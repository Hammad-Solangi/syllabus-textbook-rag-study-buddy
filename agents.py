from __future__ import annotations

import json
import os
import re
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task

from rag import SearchHit


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from environment variables or Streamlit secrets."""

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


def create_llm() -> LLM:
    """Create the configured LLM."""

    provider = get_secret(
        "LLM_PROVIDER",
        "groq",
    ).lower()

    if provider == "gemini":

        api_key = get_secret("GEMINI_API_KEY")

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

    # Default: Groq

    api_key = get_secret("GROQ_API_KEY")

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


def format_evidence(hits: list[SearchHit]) -> str:
    """Convert retrieved passages into a prompt-friendly format."""

    sections = []

    for number, hit in enumerate(hits, start=1):

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


def extract_json(text: str) -> dict[str, Any]:
    """Safely extract a JSON object from an LLM response."""

    text = text.strip()

    # Remove ```json ... ``` if the model adds Markdown fences.

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

    # First attempt: entire response is JSON.

    try:

        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Second attempt: locate a JSON object inside the response.

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
# THIS IS THE FUNCTION APP.PY IMPORTS
# ============================================================

def run_study_crew(
    question: str,
    hits: list[SearchHit],
) -> dict[str, Any]:

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create LLM
    # --------------------------------------------------------

    llm = create_llm()

    evidence = format_evidence(hits)

    # --------------------------------------------------------
    # AGENT 1
    # --------------------------------------------------------

    document_retriever = Agent(

        role="Document Retriever",

        goal=(
            "Analyze the supplied textbook evidence and identify "
            "the passages and facts that directly answer the "
            "student's question."
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

    # --------------------------------------------------------
    # AGENT 2
    # --------------------------------------------------------

    ai_tutor = Agent(

        role="AI Tutor",

        goal=(
            "Explain the answer to the student clearly using "
            "only the evidence provided by the Document Retriever."
        ),

        backstory=(
            "You are a patient textbook tutor. Your job is to "
            "turn verified textbook evidence into an easy-to-"
            "understand explanation."
        ),

        llm=llm,

        allow_delegation=False,

        verbose=False,
    )

    # --------------------------------------------------------
    # AGENT 1 TASK
    # --------------------------------------------------------

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
5. If the evidence is insufficient, say so.
""",

        expected_output=(
            "A concise evidence summary containing only "
            "facts supported by the supplied document passages."
        ),

        agent=document_retriever,
    )

    # --------------------------------------------------------
    # AGENT 2 TASK
    # --------------------------------------------------------

    tutor_task = Task(

        description=f"""
Student question:

{question}


Use the Document Retriever's evidence summary to answer
the student's question.

IMPORTANT RULES:

- Use only information supported by the retrieved document.
- Do not invent facts.
- Do not add unrelated outside knowledge.
- Explain the concept at a student-friendly level.
- If the evidence does not contain enough information,
  set "not_found" to true.
- Include source/page references when available.


Return ONLY valid JSON using this structure:

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

        context=[retrieval_task],
    )

    # --------------------------------------------------------
    # CREW
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    result = crew.kickoff()

    raw_response = getattr(
        result,
        "raw",
        str(result),
    )

    parsed = extract_json(
        raw_response
    )

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    answer = str(
        parsed.get(
            "answer",
            "",
        )
    ).strip()

    if not answer:
        answer = raw_response.strip()

    if not answer:
        answer = "The AI tutor did not return an answer."

    # --------------------------------------------------------
    # Not found
    # --------------------------------------------------------

    not_found = bool(
        parsed.get(
            "not_found",
            False,
        )
    )

    # --------------------------------------------------------
    # Citations
    # --------------------------------------------------------

    citations = parsed.get(
        "citations",
        [],
    )

    if not isinstance(
        citations,
        list,
    ):
        citations = []

    # If the model didn't provide citations,
    # generate them from retrieved evidence.

    if not citations:

        citations = sorted(
            {
                hit.source_label
                for hit in hits
            }
        )

    # --------------------------------------------------------
    # Agent 1 output
    # --------------------------------------------------------

    try:

        retriever_summary = (
            retrieval_task.output.raw
        )

    except Exception:

        retriever_summary = str(
            retrieval_task.output
        )

    # --------------------------------------------------------
    # Return result to app.py
    # --------------------------------------------------------

    return {
        "answer": answer,
        "not_found": not_found,
        "citations": citations,
        "retriever_summary": retriever_summary,
    }
