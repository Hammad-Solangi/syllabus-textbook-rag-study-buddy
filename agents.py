from __future__ import annotations

import json
import os
import re
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from rag import SearchHit

# ============================================================
# GROQ + CREWAI / LITELLM COMPATIBILITY PATCH
# ============================================================
#
# CrewAI may add "cache_breakpoint" to message dictionaries.
# Groq rejects this property in its OpenAI-compatible API.
#
# We remove the property immediately before LiteLLM sends
# the request to the provider.
# ============================================================

import litellm


_original_litellm_completion = litellm.completion


def _groq_safe_completion(*args, **kwargs):
    """
    Remove CrewAI's unsupported cache_breakpoint field
    before sending messages through LiteLLM.
    """

    messages = kwargs.get("messages")

    if isinstance(messages, list):

        cleaned_messages = []

        for message in messages:

            if isinstance(message, dict):

                message = dict(message)

                message.pop(
                    "cache_breakpoint",
                    None,
                )

            cleaned_messages.append(message)

        kwargs["messages"] = cleaned_messages

    return _original_litellm_completion(
        *args,
        **kwargs,
    )


litellm.completion = _groq_safe_completion


# ============================================================
# SECRET MANAGEMENT
# ============================================================


def get_secret(
    name: str,
    default: str = "",
) -> str:
    """
    Read a configuration value from:

    1. Environment variables
    2. Streamlit Secrets

    Environment variables take priority.
    """

    value = os.environ.get(name)

    if value:
        return value

    try:

        import streamlit as st

        value = st.secrets.get(name)

        if value:
            return str(value)

    except Exception:
        pass

    return default


# ============================================================
# LLM CREATION
# ============================================================


def create_llm() -> LLM:
    """
    Create the LLM configured in Streamlit Secrets.

    Supported providers:

    - Groq
    - Gemini
    """

    provider = get_secret(
        "LLM_PROVIDER",
        "groq",
    ).lower()

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    if provider == "gemini":

        api_key = get_secret(
            "GEMINI_API_KEY"
        )

        if not api_key:

            raise RuntimeError(
                "GEMINI_API_KEY is missing "
                "from Streamlit Secrets."
            )

        return LLM(
            model=get_secret(
                "GEMINI_MODEL",
                "gemini/gemini-2.5-flash",
            ),
            api_key=api_key,
            temperature=0.1,
            max_tokens=1200,
        )

    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    api_key = get_secret(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is missing "
            "from Streamlit Secrets."
        )

    return LLM(
        model=get_secret(
            "GROQ_MODEL",
            "groq/openai/gpt-oss-120b",
        ),
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_tokens=1200,
    )


# ============================================================
# FORMAT RETRIEVED EVIDENCE
# ============================================================


def format_evidence(
    hits: list[SearchHit],
) -> str:
    """
    Convert retrieved document chunks into a structured
    prompt for the CrewAI agents.
    """

    sections = []

    for number, hit in enumerate(
        hits,
        start=1,
    ):

        if hit.page is not None:

            location = (
                f"Page {hit.page}"
            )

        else:

            location = "Document text"

        sections.append(
            f"""
[EVIDENCE {number}]

Source: {hit.source_name}

Location: {location}

Similarity: {hit.score:.3f}

Text:
{hit.text}
""".strip()
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# JSON EXTRACTION
# ============================================================


def extract_json(
    text: str,
) -> dict[str, Any]:
    """
    Safely extract a JSON object from the AI tutor response.

    Handles:

    - Normal JSON
    - ```json fenced JSON
    - JSON embedded inside explanatory text
    """

    text = text.strip()

    # --------------------------------------------------------
    # Remove Markdown code fences
    # --------------------------------------------------------

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    # --------------------------------------------------------
    # Attempt 1: Entire response is JSON
    # --------------------------------------------------------

    try:

        result = json.loads(
            text
        )

        if isinstance(
            result,
            dict,
        ):

            return result

    except json.JSONDecodeError:

        pass

    # --------------------------------------------------------
    # Attempt 2: Find JSON object inside response
    # --------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if match:

        try:

            result = json.loads(
                match.group(0)
            )

            if isinstance(
                result,
                dict,
            ):

                return result

        except json.JSONDecodeError:

            pass

    return {}


# ============================================================
# MAIN CREWAI WORKFLOW
# ============================================================


def run_study_crew(
    question: str,
    hits: list[SearchHit],
) -> dict[str, Any]:
    """
    Run the two-agent Study Buddy workflow.

    Agent 1:
        Document Retriever

    Agent 2:
        AI Tutor
    """

    # ========================================================
    # NO RETRIEVED EVIDENCE
    # ========================================================

    if not hits:

        return {
            "answer": (
                "NOT FOUND IN DOCUMENT: "
                "No relevant evidence was retrieved."
            ),

            "not_found": True,

            "citations": [],

            "retriever_summary": (
                "The retrieval system did not find "
                "relevant passages."
            ),
        }

    # ========================================================
    # CREATE LLM
    # ========================================================

    llm = create_llm()

    # ========================================================
    # FORMAT DOCUMENT EVIDENCE
    # ========================================================

    evidence = format_evidence(
        hits
    )

    # ========================================================
    # AGENT 1 — DOCUMENT RETRIEVER
    # ========================================================

    document_retriever = Agent(

        role="Document Retriever",

        goal=(
            "Analyze the supplied textbook evidence "
            "and identify the passages and facts that "
            "directly answer the student's question."
        ),

        backstory=(
            "You are a careful academic research assistant. "
            "You must work ONLY with the document evidence "
            "provided to you. Never invent information."
        ),

        llm=llm,

        allow_delegation=False,

        verbose=False,
    )

    # ========================================================
    # AGENT 2 — AI TUTOR
    # ========================================================

    ai_tutor = Agent(

        role="AI Tutor",

        goal=(
            "Explain the answer to the student clearly "
            "using only the evidence provided by the "
            "Document Retriever."
        ),

        backstory=(
            "You are a patient textbook tutor. "
            "Your job is to turn verified textbook "
            "evidence into an easy-to-understand "
            "explanation."
        ),

        llm=llm,

        allow_delegation=False,

        verbose=False,
    )

    # ========================================================
    # TASK 1 — RETRIEVE / VERIFY EVIDENCE
    # ========================================================

    retrieval_task = Task(

        description=f"""
Student question:

{question}


Retrieved textbook evidence:

{evidence}


Your job:

1. Identify which passages are relevant.

2. Extract the facts needed to answer the question.

3. Identify the source and page where possible.

4. Do not use outside knowledge.

5. Determine whether the evidence provides a direct,
   partial, or no answer to the question.

6. If the evidence is partial, identify exactly which
   facts are supported and which information is missing.

7. Do not invent or reconstruct information that is
   not supported by the supplied evidence.
""",

        expected_output=(
            "A concise evidence summary containing only "
            "facts supported by the supplied document "
            "passages."
        ),

        agent=document_retriever,
    )

    # ========================================================
    # TASK 2 — TEACH THE STUDENT
    # ========================================================

       tutor_task = Task(

        description=f"""
Student question:

{question}


Use the Document Retriever's evidence summary to answer
the student's question.

IMPORTANT RULES:

1. Use the retrieved document evidence as the primary
   source of truth.

2. Do NOT invent facts that contradict the document.

3. If the document directly answers the question,
   provide a clear student-friendly answer.

4. If the document provides PARTIAL information about
   the question, answer using ONLY the supported
   information and clearly state what the document does
   and does not explain.

5. If the document mentions the requested concept but
   does not provide a complete definition, do NOT mark
   it as NOT FOUND. Instead, explain the information that
   IS present in the document.

6. Only set "not_found" to true when the retrieved
   evidence contains no meaningful information relevant
   to the question.

7. For mathematics, show the relevant solution steps when
   those steps are supported by the supplied material.

8. Include source/page references when available.

9. Do not pretend that information exists in the document
   when it does not.

Student-friendly explanations are encouraged, but any
information presented as coming from the textbook must be
supported by the retrieved evidence.


Return ONLY valid JSON using exactly this structure:

{{
    "answer": "Your student-friendly answer",
    "not_found": false,
    "citations": [
        "filename.pdf — Page 4"
    ]
}}
""",

        expected_output=(
            "Valid JSON containing answer, "
            "not_found, and citations."
        ),

        agent=ai_tutor,

        context=[
            retrieval_task
        ],
    )
    # ========================================================
    # CREATE CREW
    # ========================================================

    crew = Crew(

        agents=[
            document_retriever,
            ai_tutor,
        ],

        tasks=[
            retrieval_task,
            tutor_task,
        ],

        process=Process.sequential,

        verbose=False,
    )

    # ========================================================
    # RUN CREW
    # ========================================================

    result = crew.kickoff()

    # ========================================================
    # GET RAW RESPONSE
    # ========================================================

    raw_response = getattr(
        result,
        "raw",
        str(result),
    )

    if raw_response is None:

        raw_response = ""

    raw_response = str(
        raw_response
    ).strip()

    # ========================================================
    # PARSE JSON
    # ========================================================

    parsed = extract_json(
        raw_response
    )

    # ========================================================
    # ANSWER
    # ========================================================

    answer = str(
        parsed.get(
            "answer",
            "",
        )
    ).strip()

    if not answer:

        answer = raw_response

    if not answer:

        answer = (
            "The AI tutor did not return an answer."
        )

    # ========================================================
    # NOT FOUND
    # ========================================================

    not_found = bool(
        parsed.get(
            "not_found",
            False,
        )
    )

    # ========================================================
    # CITATIONS
    # ========================================================

    citations = parsed.get(
        "citations",
        [],
    )

    if not isinstance(
        citations,
        list,
    ):

        citations = []

    # If the AI tutor didn't return citations,
    # generate them from the retrieved evidence.

    if not citations:

        citations = sorted(
            {
                hit.source_label
                for hit in hits
            }
        )

    # ========================================================
    # AGENT 1 SUMMARY
    # ========================================================

    try:

        retriever_summary = (
            retrieval_task.output.raw
        )

    except Exception:

        try:

            retriever_summary = str(
                retrieval_task.output
            )

        except Exception:

            retriever_summary = (
                "The Document Retriever "
                "completed without a readable "
                "summary."
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "answer": answer,

        "not_found": not_found,

        "citations": citations,

        "retriever_summary": retriever_summary,
    }
