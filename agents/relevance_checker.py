from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.constants import TOP_K

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

relevance_checker_prompt = ChatPromptTemplate.from_template(
    """
    You are an AI relevance checker between a user's question and provided document content.
    - Classify how well the document content addresses the user's question.
    - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_ANSWER.
    - Do not include any additional text or explanation.
    **Labels:**
    1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
    2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
    3) "NO_ANSWER": The passages do not discuss or mention the question's topic at all.
    **Question:** {question}
    **Passages:** {document_content}
    **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_ANSWER**
    """
)

relevance_checker_pipe = relevance_checker_prompt | llm

def relevance_checker(state, retriever):
    top_docs = retriever.invoke(state.get("question", ""))
    if not top_docs:
        return {"q_type": "NO_ANSWER", "documents": []}
    document_content = "\n\n".join(doc.page_content for doc in top_docs[:TOP_K])
    response = relevance_checker_pipe.invoke({
        "question": state.get("question", ""),
        "document_content": document_content
    })
    return {"q_type": response.content.upper(), "documents": top_docs}
