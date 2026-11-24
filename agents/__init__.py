"""The specialist agents wired together by graph.py.

Each module exposes a make_*_node(...) factory that closes over the
shared resources it needs (LLM, retriever) and returns a LangGraph node.
Nodes return only the state keys they change.
"""
