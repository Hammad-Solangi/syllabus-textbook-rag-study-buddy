# 📚 Syllabus & Textbook RAG Study Buddy

An AI-powered study assistant built with:

- Streamlit
- CrewAI
- Groq or Gemini
- PDF/DOCX parsing
- TF-IDF vector retrieval
- Source-grounded question answering

The application allows a student to upload their syllabus, textbook, notes, or other study material and ask questions about it.

---

# Architecture

```text
                 ┌──────────────────┐
                 │   Streamlit UI   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  PDF / DOCX     │
                 │     Parser       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   Text Chunking  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ TF-IDF Vector    │
                 │    Retrieval     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ CrewAI Agent 1   │
                 │ Document         │
                 │ Retriever        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ CrewAI Agent 2   │
                 │ AI Tutor         │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Answer + Sources │
                 └──────────────────┘
