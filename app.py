
import gradio as gr
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from docling.document_converter import DocumentConverter
import tiktoken
from retriever.retriever_builder import build_hybrid_retriever
from config.constants import TOP_K
from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langchain.schema import Document
from agents.workflow import build_workflow

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

question_type = Literal['CAN_ANSWER', 'PARTIAL', 'NO_ANSWER']
class VerifyResult(BaseModel):
    Supported: str = Field(...)
    Unsupported_Claims: List[str] = Field(default_factory=list)
    Contradictions: List[str] = Field(default_factory=list)
    Relevant: str = Field(...)
    Additional_Details: List[str] = Field(default_factory=list)

class State(TypedDict):
    question: str
    answer: str
    context: str
    q_type: question_type
    documents: List[Document]
    report: VerifyResult
    n: int

def process_pdf_to_chunks(file_path: str):
    converter = DocumentConverter()
    markdown = converter.convert(file_path).document.export_to_markdown()
    headers = [("#", "Header 1"), ("##", "Header 2")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    chunks = splitter.split_text(markdown)
    return chunks

def build_agent_workflow(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")
    MAX_TOKENS = embeddings.embedding_ctx_length
    for chunk in chunks:
        if hasattr(chunk, "page_content") and len(tokenizer.encode(chunk.page_content)) > MAX_TOKENS:
            print("Invalid chunk exceed token length")
    retriever = build_hybrid_retriever(chunks, embeddings)
    workflow = build_workflow(State)
    reflection_workflow = workflow.compile()
    from agents.relevance_checker import relevance_checker
    def relevance_checker_with_retriever(state):
        return relevance_checker(state, retriever)
    reflection_workflow.graph.nodes["check_relevance"].func = relevance_checker_with_retriever
    return reflection_workflow

def retriever_qa(file, query):
    if not file or not query:
        return "Please upload a PDF and enter a question."
    try:
        chunks = process_pdf_to_chunks(file)
        workflow = build_agent_workflow(chunks)
        state = {"question": query, "n": 0}
        response = workflow.invoke(state)
        answer = response.get("answer", "No answer found.")
        return answer
    except Exception as e:
        return f"Error: {e}"

rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Document Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document. Guarantee halluciation-free answer"
)

rag_application.launch(server_name="127.0.0.1", server_port=7870, share=True)