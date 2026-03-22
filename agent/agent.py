# agent/agent.py
# ============================================================
#  Medical AI Agent — LangGraph v0.4 + LangChain v0.3
#
#  LangGraph v0.4 key changes:
#  - StateGraph API is the same but compilation is stricter
#  - Annotated fields with operator.add for list accumulation
#  - END imported from langgraph.graph (unchanged)
#  - .compile() returns a CompiledStateGraph (same invoke API)
#
#  LangChain v0.3 key changes:
#  - langchain_groq: ChatGroq constructor is cleaner
#  - Messages: HumanMessage, SystemMessage still from langchain_core
#  - No more LLMChain — use model.invoke() directly
#
#  Flow:
#  rewrite_query → hybrid_retrieval → [route] → build_context → generate_response
#                                         ↓
#                                    web_search → build_context
# ============================================================

from typing import TypedDict, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from utils.config import config
from utils.logger import logger
from retriever.hybrid import HybridRetriever
from tools.web_search import web_search, format_web_context


# ─────────────────────────────────────────────────────────────
#  STATE — the shared data bag passed between every node
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:             str
    rewritten_query:   str
    rag_results:       list
    web_results:       list
    confidence:        float
    used_web_search:   bool
    context:           str
    response:          str
    source:            str
    chat_history:      list   # list of {"user": str, "assistant": str}


# ─────────────────────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional medical information assistant.

STRICT RULES — follow every single one:
1. Use ONLY information from the [CONTEXT] section below.
2. Do NOT invent, guess, or hallucinate any medical facts.
3. If context is insufficient, say exactly: "I don't have enough information to answer that reliably."
4. Structure your answer clearly with these sections:
   - **Possible Condition(s)**
   - **Key Symptoms Match**
   - **Recommended Treatments**
   - **Important Notes**
5. Keep a calm, professional, and empathetic tone.
6. ALWAYS end with the disclaimer below — never skip it.

⚠️ Medical Disclaimer: This information is for educational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional for proper diagnosis and treatment."""

REWRITE_PROMPT = """You are a medical search query optimizer.

Rewrite the user's query into a clean, specific medical search query.
- Under 20 words
- Use proper medical terminology
- Keep only the core symptoms/conditions
- Do NOT add information not in the original

Original: {query}

Reply with ONLY the rewritten query, nothing else."""


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _build_rag_context(results: list) -> str:
    if not results:
        return "No relevant medical records found in the internal database."

    lines = ["[DATABASE CONTEXT]\n"]
    for i, r in enumerate(results[:5], 1):
        doc = r["doc"]
        lines.append(
            f"Record {i} (Match Score: {r['score']:.2f}):\n"
            f"  Disease   : {doc['name']}\n"
            f"  Symptoms  : {doc['symptoms']}\n"
            f"  Treatment : {doc['treatments']}\n"
            f"  Contagious: {doc['contagious']} | Chronic: {doc['chronic']}\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────────────────────────

class MedicalAgent:
    """
    LangGraph v0.4 state-machine agent.

    Graph nodes:
        rewrite_query
            ↓
        hybrid_retrieval
            ↓  (conditional)
        ┌── build_context  ←── (high confidence)
        └── web_search → build_context  ←── (low confidence)
            ↓
        generate_response
            ↓
          END
    """

    def __init__(self, hybrid_retriever: HybridRetriever):
        self.retriever = hybrid_retriever

        # LangChain v0.3 + langchain-groq v0.3
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

        self.graph = self._build_graph()
        logger.info("MedicalAgent initialised with LangGraph v0.4")

    # ── Node: Query Rewriting ────────────────────────────────
    def _rewrite_query(self, state: AgentState) -> AgentState:
        try:
            prompt = REWRITE_PROMPT.format(query=state["query"])
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            rewritten = resp.content.strip()
            logger.info(f"Rewritten: '{state['query']}' → '{rewritten}'")
        except Exception as e:
            logger.warning(f"Rewrite failed ({e}), using original")
            rewritten = state["query"]
        return {**state, "rewritten_query": rewritten}

    # ── Node: Hybrid Retrieval ───────────────────────────────
    def _run_hybrid_retrieval(self, state: AgentState) -> AgentState:
        query = state.get("rewritten_query") or state["query"]
        results, confidence = self.retriever.retrieve(query)
        logger.info(f"Hybrid confidence: {confidence:.3f}")
        return {**state, "rag_results": results, "confidence": confidence}

    # ── Node: Web Search ─────────────────────────────────────
    def _run_web_search(self, state: AgentState) -> AgentState:
        logger.info("Low confidence → web search triggered")
        results = web_search(state["query"])
        return {**state, "web_results": results, "used_web_search": True}

    # ── Node: Build Context ──────────────────────────────────
    def _build_context(self, state: AgentState) -> AgentState:
        parts = []

        if state.get("rag_results"):
            parts.append(_build_rag_context(state["rag_results"]))

        if state.get("used_web_search") and state.get("web_results"):
            parts.append(
                "[WEB SEARCH CONTEXT]\n" + format_web_context(state["web_results"])
            )

        context = "\n\n".join(parts) if parts else "No context available."
        source  = "web_search" if state.get("used_web_search") else "database"
        return {**state, "context": context, "source": source}

    # ── Node: Generate Response ──────────────────────────────
    def _generate_response(self, state: AgentState) -> AgentState:
        # Build message list — system prompt first
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Inject last 4 turns of chat history for memory
        for turn in (state.get("chat_history") or [])[-4:]:
            messages.append(HumanMessage(content=turn["user"]))
            # Re-inject previous assistant reply as a human-turn context note
            # (Groq/Llama doesn't support AIMessage in system role, so we wrap it)
            messages.append(
                HumanMessage(content=f"[Previous assistant response]: {turn['assistant']}")
            )

        # Current query + context
        user_msg = (
            f"[CONTEXT]\n{state['context']}\n\n"
            f"[PATIENT QUERY]\n{state['query']}"
        )
        messages.append(HumanMessage(content=user_msg))

        try:
            resp = self.llm.invoke(messages)
            raw = resp.content.strip()
        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            raw = "I'm unable to process your query right now. Please try again."

        # Append confidence + source footer
        conf_pct      = int(state["confidence"] * 100)
        source_label  = "Web Search" if state.get("used_web_search") else "Medical Database"
        final = (
            f"{raw}\n\n"
            f"---\n"
            f"📊 Confidence Score: {conf_pct}% | Source: {source_label}"
        )

        logger.info(f"Response generated | conf={conf_pct}% | source={source_label}")
        return {**state, "response": final}

    # ── Conditional router ───────────────────────────────────
    def _route_after_retrieval(self, state: AgentState) -> str:
        """Return next node name based on confidence."""
        if self.retriever.is_confident(state["confidence"]):
            logger.info("Confidence OK → using RAG context")
            return "build_context"
        logger.info("Confidence LOW → web search")
        return "web_search"

    # ── Build the LangGraph ──────────────────────────────────
    def _build_graph(self) -> "CompiledStateGraph":
        """
        Assemble the LangGraph v0.4 state machine.

        LangGraph v0.4 note:
        - StateGraph(AgentState) — same as before
        - add_node / add_edge / add_conditional_edges — same API
        - set_entry_point — same
        - compile() — same, returns CompiledStateGraph
        """
        g = StateGraph(AgentState)

        # Register nodes
        g.add_node("rewrite_query",     self._rewrite_query)
        g.add_node("hybrid_retrieval",  self._run_hybrid_retrieval)
        g.add_node("web_search",        self._run_web_search)
        g.add_node("build_context",     self._build_context)
        g.add_node("generate_response", self._generate_response)

        # Entry point
        g.set_entry_point("rewrite_query")

        # Fixed edges
        g.add_edge("rewrite_query",     "hybrid_retrieval")
        g.add_edge("web_search",        "build_context")
        g.add_edge("build_context",     "generate_response")
        g.add_edge("generate_response", END)

        # Conditional edge after retrieval
        g.add_conditional_edges(
            "hybrid_retrieval",
            self._route_after_retrieval,
            {
                "build_context": "build_context",
                "web_search":    "web_search",
            },
        )

        return g.compile()

    # ── Public API ───────────────────────────────────────────
    def run(self, query: str, chat_history: list = None) -> dict:
        """
        Process a user query end-to-end.

        Args:
            query        : Symptom / medical question string
            chat_history : Optional list of past turns

        Returns dict with keys:
            response, confidence, source, used_web_search, rag_results
        """
        initial: AgentState = {
            "query":           query,
            "rewritten_query": "",
            "rag_results":     [],
            "web_results":     [],
            "confidence":      0.0,
            "used_web_search": False,
            "context":         "",
            "response":        "",
            "source":          "",
            "chat_history":    chat_history or [],
        }

        logger.info(f"Agent processing: '{query}'")
        final = self.graph.invoke(initial)

        return {
            "response":        final["response"],
            "confidence":      final["confidence"],
            "source":          final["source"],
            "used_web_search": final["used_web_search"],
            "rag_results": [
                {"name": r["doc"]["name"], "score": round(r["score"], 3)}
                for r in final["rag_results"][:3]
            ],
        }
