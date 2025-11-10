from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3, max_tokens=300)
generate_draft_answer = ChatPromptTemplate.from_template(
    """
    You are an AI assistant designed to provide precise and factual answers based on the given context.
    - Answer the following question using only the provided context.
    - Be clear, concise, and factual.
    - Return as much information as you can get from the context.
    **Question:** {question}
    **Context:**
    {context}
    **Provide your answer below:**
    """
)
generate_draft_pipe = generate_draft_answer | llm

def generate_draft(state):
    documents = state.get("documents")
    context = "\n\n".join([doc.page_content for doc in documents]) if documents else ""
    response = generate_draft_pipe.invoke({
        "question": state.get("question", ""),
        "context": context
    })
    llm_response = response.content.strip()
    return {"answer": llm_response, "context": context}
