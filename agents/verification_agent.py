from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

class VerifyResult(BaseModel):
    Supported: str = Field(...)
    Unsupported_Claims: List[str] = Field(default_factory=list)
    Contradictions: List[str] = Field(default_factory=list)
    Relevant: str = Field(...)
    Additional_Details: List[str] = Field(default_factory=list)

llm = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=300)
verify_answer_prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.
    - Verify the following answer against the provided context.
    - Check for:
    1. Direct/indirect factual support (YES/NO)
    2. Unsupported claims (list any if present)
    3. Contradictions (list any if present)
    4. Relevance to the question (YES/NO)
    - Provide additional details or explanations where relevant.
    - Respond in the exact format specified below without adding any unrelated information.
    **Format:**
    Supported: YES/NO
    Unsupported Claims: [item1, item2, ...]
    Contradictions: [item1, item2, ...]
    Relevant: YES/NO
    Additional Details: [Any extra information or explanations]
    **Answer:** {answer}
    **Context:**
    {context}
    **Respond ONLY with the above format.**
    """
)
verify_answer_pipe = verify_answer_prompt | llm.with_structured_output(VerifyResult)

def verify_check(state):
    documents = state.get("documents")
    context = "\n\n".join([doc.page_content for doc in documents]) if documents else ""
    response = verify_answer_pipe.invoke({
        "answer": state.get("answer", ""),
        "context": context
    })
    verification_report_formatted = (
        f"Supported: {response.Supported if response.Supported else 'NO'}\n"
        f"Unsupported Claims: {response.Unsupported_Claims if response.Unsupported_Claims else []}\n"
        f"Contradictions: {response.Contradictions if response.Contradictions else []}\n"
        f"Relevant: {response.Relevant if response.Relevant else 'NO'}\n"
        f"Additional Details: {response.Additional_Details if response.Additional_Details else []}\n"
    )
    return {"report": verification_report_formatted}
