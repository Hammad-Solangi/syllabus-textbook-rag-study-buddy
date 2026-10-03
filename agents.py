from __future__ import annotations

import json
import os
import re
from typing import Any

import litellm

from crewai import Agent, Crew, LLM, Process, Task
from rag import SearchHit


# ============================================================
# GROQ / LITELLM COMPATIBILITY
# ============================================================

_original_litellm_completion = litellm.completion


def _groq_safe_completion(*args, **kwargs):
    """
    Remove cache_breakpoint from messages before they are
    sent to Groq.
    """

    messages = kwargs.get("messages")

    if isinstance(messages, list):
        cleaned_messages = []

        for message in messages:
            if isinstance(message, dict):
                message = dict(message)
                message.pop("cache_breakpoint", None)

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
    """Read configuration from environment or Streamlit Secrets."""

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
# LLM
# ============================================================

def create_llm() -> LLM:
    """Create the configured language model."""

    provider = get_secret(
        "LLM_PROVIDER",
        "groq",
    ).lower()

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    if provider == "gemini":

        api_key = get_secret(
            "GEMINI_API_KEY",
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing from Streamlit Secrets."
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
    # Groq
    # --------------------------------------------------------

    api_key = get_secret(
        "GROQ_API_KEY",
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
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
# EVIDENCE FORMATTING
# ============================================================

def format_evidence(
    hits: list[SearchHit],
) -> str:
    """Format retrieved document chunks for the agents."""

    sections = []

    for number, hit in enumerate(
        hits,
        start=1,
    ):

        if hit.page is not None:
            location = f"Page {hit.page}"
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

    return "\n\n".join(sections)


# ============================================================
# JSON PARSING
# ============================================================

def extract_json(
    text: str,
) -> dict[str, Any]:
    """Extract JSON safely from an LLM response."""

    text = text.strip()

    # Remove Markdown JSON fences.

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

    # Try complete response as JSON.

    try:

        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Try finding JSON inside the response.

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

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    return {}


# ============================================================
# MAIN STUDY CREW
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
    # NO EVIDENCE
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
    # LLM
    # ========================================================

    llm = create_llm()

    evidence = format_evidence(
        hits
    )

    # ========================================================
    # AGENT 1 — DOCUMENT RETRIEVER
    # ========================================================

    document_retriever = Agent(
        role="Document Retriever",

        goal=(
            "Analyze the supplied textbook evidence and "
            "identify the passages and facts that answer "
            "the student's question."
        ),

        backstory=(
            "You are a careful academic research assistant. "
            "You work ONLY with the document evidence provided "
            "to you. You never invent information."
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
            "using the evidence provided by the Document "
            "Retriever."
        ),

        backstory=(
            "You are a patient textbook tutor. You turn "
            "verified textbook evidence into clear, "
            "student-friendly explanations."
        ),

        llm=llm,

        allow_delegation=False,

        verbose=False,
    )

    # ========================================================
    # TASK 1 — DOCUMENT ANALYSIS
    # ========================================================

    retrieval_task = Task(
        description=f"""
Student question:

{question}


Retrieved textbook evidence:

{evidence}


Analyze the evidence carefully.

Your job:

1. Identify which passages are relevant.

2. Extract the facts needed to answer the question.

3. Identify the source and page where possible.

4. Do not use outside knowledge.

5. Decide whether the evidence provides:
   - a direct answer,
   - a partial answer,
   - or no meaningful answer.

6. If the evidence is partial, clearly identify what
   information IS supported by the document.

7. Do not invent or reconstruct information that is
   not supported by the supplied evidence.
""",

        expected_output=(
            "A concise evidence analysis explaining whether "
            "the retrieved material directly, partially, or "
            "not at all answers the student's question."
        ),

        agent=document_retriever,
    )

    # ========================================================
    # TASK 2 — AI TUTOR
    # ========================================================

    tutor_task = Task(
        description=f"""
Student question:

{question}


Use the Document Retriever's evidence analysis to answer
the student's question.

IMPORTANT RULES:

1. Use the retrieved document evidence as the primary
   source of truth.

2. Do not invent facts.

3. Do not contradict the document.

4. If the document directly answers the question,
   provide a clear student-friendly answer.

5. If the document provides PARTIAL information,
   answer using the information that IS supported.

6. If the document mentions the requested concept but
   does not provide a complete definition, do NOT mark
   it as NOT FOUND. Explain what the document actually
   says and clearly mention what is missing.

7. Only set "not_found" to true when the retrieved
   evidence contains no meaningful information relevant
   to the question.

8. For mathematics, show relevant solution steps when
   those steps are supported by the document.

9. Include source/page references whenever available.

10. Do not claim that something is written in the
    textbook if it is not supported by the evidence.


Return ONLY valid JSON in this exact structure:

{{
    "answer": "Your student-friendly answer",
    "not_found": false,
    "citations": [
        "filename.pdf — Page 4"
    ]
}}
""",

        expected_output=(
            "Valid JSON containing answer, not_found, "
            "and citations."
        ),

        agent=ai_tutor,

        context=[
            retrieval_task,
        ],
    )

    # ========================================================
    # CREW
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
    # EXECUTE
    # ========================================================

    result = crew.kickoff()

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
    # PARSE ANSWER
    # ========================================================

    parsed = extract_json(
        raw_response
    )

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
                "The Document Retriever completed "
                "without a readable summary."
            )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "answer": answer,
        "not_found": not_found,
        "citations": citations,
        "retriever_summary": retriever_summary,
    }
