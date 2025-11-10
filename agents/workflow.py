from langgraph.graph import StateGraph, END
from config.constants import TOP_K
from agents.relevance_checker import relevance_checker
from agents.research_agent import generate_draft
from agents.verification_agent import verify_check

# Routing functions

def check_relevance(state):
    question_type = state.get("q_type", "NO_ANSWER")
    if question_type == 'CAN_ANSWER' or question_type == 'PARTIAL':
        return "relevance"
    if question_type == 'NO_ANSWER':
        return "irrelevant"

def route_research(state):
    verification_report = state.get("report")
    if verification_report:
        if "Supported: NO" in verification_report or "Relevant: NO" in verification_report:
            return "re_research"
        else:
            return "end"
    else:
        return "end"

def build_workflow(State):
    workflow = StateGraph(State)
    workflow.add_node("check_relevance", relevance_checker)
    workflow.add_node("research", generate_draft)
    workflow.add_node("verify", verify_check)
    workflow.add_node("irrelevant_handler", lambda state: {
        **state,
        "answer": "This question isn't related (or there's no data) for your query. Please ask related question."
    })
    workflow.set_entry_point("check_relevance")
    workflow.add_conditional_edges(
        "check_relevance",
        check_relevance,
        {
            "relevance": "research",
            "irrelevant": "irrelevant_handler"
        }
    )
    workflow.add_edge("irrelevant_handler", END)
    workflow.add_edge("research", "verify")
    workflow.add_conditional_edges(
        "verify",
        route_research,
        {
            "re_research": "research",
            "end": END
        }
    )
    return workflow
