from langgraph.graph import END, START, StateGraph

from . import nodes
from .nodes import SupportState


def build_graph():
    builder = StateGraph(SupportState)
    for name, fn in [("condense_question", nodes.condense_question), ("classify_intent", nodes.classify_intent),
                     ("clarify", nodes.clarify), ("retrieve", nodes.retrieve),
                     ("draft_answer", nodes.draft_answer), ("review_answer", nodes.review_answer),
                     ("rewrite_query", nodes.rewrite_query), ("fallback", nodes.fallback),
                     ("check_coverage", nodes.check_coverage)]:
        builder.add_node(name, fn)

    builder.add_edge(START, "condense_question")
    builder.add_edge("condense_question", "classify_intent")
    builder.add_conditional_edges("classify_intent", nodes.route_after_classify,
                                  {"retrieve": "retrieve", "clarify": "clarify", END: END})
    builder.add_edge("clarify", END)
    builder.add_edge("retrieve", "draft_answer")
    builder.add_edge("draft_answer", "review_answer")
    builder.add_conditional_edges("review_answer", nodes.route_after_review,
                                  {"rewrite_query": "rewrite_query", "fallback": "fallback",
                                   "check_coverage": "check_coverage"})
    builder.add_edge("check_coverage", END)
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("fallback", END)
    return builder.compile()   # no checkpointer: memory is handled by SessionStore
