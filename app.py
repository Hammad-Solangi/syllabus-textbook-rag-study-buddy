import hashlib
from typing import Any

import streamlit as st

from agents import run_study_crew
from rag import DocumentIndex, SUPPORTED_EXTENSIONS


st.set_page_config(
    page_title="Syllabus & Textbook RAG Study Buddy",
    page_icon="📚",
    layout="wide",
)


st.title("📚 Syllabus & Textbook RAG Study Buddy")

st.caption(
    "Ask questions from your uploaded textbook, notes, or syllabus. "
    "Answers are grounded in retrieved document passages."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Retrieved passages",
        min_value=2,
        max_value=8,
        value=5,
    )

    st.markdown(
        """
### How it works

1. Upload PDF/DOCX study material.
2. The app extracts the text.
3. Relevant passages are retrieved.
4. CrewAI Agent 1 analyzes the evidence.
5. CrewAI Agent 2 explains it as a tutor.
6. Sources/pages are shown with the answer.

If the material does not contain enough evidence,
the tutor should say:

**NOT FOUND IN DOCUMENT**
"""
    )


# ---------------------------------------------------------
# File uploader
# ---------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload your study material",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True,
)


if not uploaded_files:
    st.info("Upload one or more PDF/DOCX files to begin.")
    st.stop()


# ---------------------------------------------------------
# Generate a signature for uploaded files
# ---------------------------------------------------------

signature_hash = hashlib.sha256()

for uploaded in uploaded_files:
    signature_hash.update(uploaded.name.encode("utf-8"))
    signature_hash.update(uploaded.getvalue())

signature = signature_hash.hexdigest()


# ---------------------------------------------------------
# Build/rebuild document index
# ---------------------------------------------------------

if st.session_state.get("index_signature") != signature:

    with st.spinner("📖 Reading and indexing your study material..."):

        try:
            index = DocumentIndex.from_uploaded_files(
                uploaded_files
            )

            st.session_state.document_index = index
            st.session_state.index_signature = signature
            st.session_state.last_answer = None

        except Exception as exc:

            st.error(
                f"Could not index the uploaded material: {exc}"
            )

            st.stop()


index: DocumentIndex = st.session_state.document_index


# ---------------------------------------------------------
# Document information
# ---------------------------------------------------------

col1, col2 = st.columns([2, 1])

with col1:

    st.success(
        f"Indexed **{index.document_count}** document(s), "
        f"**{index.chunk_count}** searchable passage(s)."
    )

with col2:

    st.metric(
        "Documents",
        index.document_count,
    )


with st.expander("📄 Uploaded material"):

    for name in index.document_names:
        st.write(f"- {name}")


# ---------------------------------------------------------
# Question
# ---------------------------------------------------------

st.divider()

st.subheader("Ask your study question")

question = st.text_area(
    "Question",
    placeholder=(
        "Example: What were the main ideas of Rutherford's atomic model?"
    ),
    height=110,
    label_visibility="collapsed",
)


# ---------------------------------------------------------
# Ask button
# ---------------------------------------------------------

if st.button(
    "🔎 Ask Study Buddy",
    type="primary",
    use_container_width=True,
):

    if not question.strip():

        st.warning("Please enter a question.")
        st.stop()


    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    with st.spinner("🔍 Retrieving relevant passages..."):

        hits = index.search(
            question,
            top_k=top_k,
        )


    if not hits:

        st.warning(
            "NOT FOUND IN DOCUMENT: "
            "I couldn't find relevant evidence "
            "in the uploaded material."
        )

        st.stop()


    # -----------------------------------------------------
    # Show retrieved evidence
    # -----------------------------------------------------

    with st.expander(
        "🔍 Retrieved evidence",
        expanded=False,
    ):

        for i, hit in enumerate(hits, start=1):

            st.markdown(
                f"**{i}. {hit.source_label}**  \n"
                f"Similarity: `{hit.score:.3f}`"
            )

            st.write(hit.text)


    # -----------------------------------------------------
    # CrewAI
    # -----------------------------------------------------

    with st.spinner(
        "🤖 Two-agent tutor is preparing your answer..."
    ):

        try:

            result = run_study_crew(
                question=question.strip(),
                hits=hits,
            )

        except Exception as exc:

            st.error(
                "The AI tutor could not complete the request. "
                "Check your API key and deployment logs."
            )

            st.exception(exc)

            st.stop()


    st.session_state.last_answer = result


# ---------------------------------------------------------
# Display answer
# ---------------------------------------------------------

result: dict[str, Any] | None = st.session_state.get(
    "last_answer"
)


if result:

    st.divider()

    st.subheader("🤖 Study Buddy Answer")


    if result.get("not_found"):

        st.warning(
            "NOT FOUND IN DOCUMENT"
        )


    st.markdown(
        result.get(
            "answer",
            "No answer was returned.",
        )
    )


    # -----------------------------------------------------
    # Citations
    # -----------------------------------------------------

    citations = result.get(
        "citations",
        [],
    )


    if citations:

        st.subheader("📚 Sources")

        for citation in citations:

            st.markdown(
                f"- {citation}"
            )


    # -----------------------------------------------------
    # Agent 1 output
    # -----------------------------------------------------

    with st.expander(
        "🧠 Agent 1 — Evidence Summary"
    ):

        st.write(
            result.get(
                "retriever_summary",
                "No summary returned.",
            )
        )


    st.caption(
        "Source-grounded answers can still contain model errors. "
        "Verify important information against the cited passages."
    )
