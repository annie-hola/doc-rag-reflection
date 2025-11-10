# doc-rag-reflection

## About the project

doc-rag-reflection is a multi-agent Retrieval-Augmented Generation (RAG) system designed to answer questions about documents while minimizing hallucinations using a reflection pattern. The system processes uploaded documents, breaks them into semantic chunks, performs hybrid retrieval (BM25 + vector search), and uses a multi-agent LangGraph workflow to generate, verify, and refine answers.

Key goals:
- Provide answers grounded in the source documents.
- Reduce hallucinations via an automated verification/reflection loop.
- Support PDF documents converted to Markdown and split into logical chunks.

