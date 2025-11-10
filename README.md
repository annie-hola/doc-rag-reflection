# doc-rag-reflection

## About the project

doc-rag-reflection is a multi-agent Retrieval-Augmented Generation (RAG) system designed to answer questions about documents while minimizing hallucinations using a reflection pattern. The system processes uploaded documents, breaks them into semantic chunks, performs hybrid retrieval (BM25 + vector search), and uses a multi-agent LangGraph workflow to generate, verify, and refine answers.


Key goals:
- Provide answers grounded in the source documents.
- Reduce hallucinations via an automated verification/reflection loop.
- Support PDF documents converted to Markdown and split into logical chunks.

## Python Environment Setup

It is recommended to use a virtual environment for Python:

Create and activate a virtual environment:

	```sh
	python3 -m venv .venv
	source .venv/bin/activate
	```


## Environment Variables

Before running the application, create a `.env` file in the project root with the following keys:

```env
OPENAI_API_KEY=your_openai_api_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_BASE_URL=your_langfuse_base_url
```

Replace each value with your actual API credentials.

Continue with the run instructions below.

## How to Run the Application

1. Install dependencies:

	```sh
	make install
	```
> For MacOS system, to fix `pygraphviz` build error. Install Graphviz system dependencies first: `brew install graphviz`

2. Run the application:

	```sh
	make run
	```

3. Open your browser and go to [http://127.0.0.1:7870](http://127.0.0.1:7870) to use the Document Chatbot.

## Step-by-Step Details

For a detailed, step-by-step explanation of the workflow and agent logic, see the notebook:

- `research/rag-reflection-docpdf.ipynb`

This notebook demonstrates the full process, including document processing, retrieval, agent orchestration, and verification.

