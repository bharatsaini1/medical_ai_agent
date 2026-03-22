# api/app.py
# ============================================================
#  Flask REST API — Medical AI Agent
#
#  Endpoints:
#    GET  /              — serves the frontend UI
#    GET  /health        — health check
#    POST /chat          — main chat endpoint
#    POST /reset         — clear chat history for a session
#    GET  /sessions      — debug: list active sessions
# ============================================================

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from utils.config import config
from utils.logger import logger
from utils.data_processor import load_documents
from retriever.hybrid import HybridRetriever
from agent.agent import MedicalAgent


# ─────────────────────────────────────────────────────────────
#  STARTUP — initialise all components once
# ─────────────────────────────────────────────────────────────

logger.info("Starting Medical AI Agent server...")

try:
    config.validate()
    logger.info("API keys validated ✓")
except EnvironmentError as e:
    logger.error(str(e))
    sys.exit(1)

logger.info("Loading medical dataset...")
documents = load_documents()
logger.info(f"Dataset loaded: {len(documents)} medical records")

logger.info("Initialising Hybrid Retriever...")
hybrid_retriever = HybridRetriever()
hybrid_retriever.setup(documents)
logger.info("Hybrid Retriever ready ✓")

logger.info("Initialising AI Agent...")
agent = MedicalAgent(hybrid_retriever)
logger.info("AI Agent ready ✓")


# ── Background indexing ──────────────────────────────────────
# Run Pinecone indexing in background so Flask starts immediately
# Health check passes right away, indexing completes within 2 min
import threading

def background_setup():
    try:
        logger.info("Background: Starting Pinecone indexing...")
        hybrid_retriever.vector_retriever.index_documents(documents)
        logger.info("Background: Pinecone indexing complete ✓")
    except Exception as e:
        logger.error(f"Background indexing error: {e}")

indexing_thread = threading.Thread(target=background_setup, daemon=True)
indexing_thread.start()
logger.info("Background indexing thread started — Flask starting now")

# ─────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────

# Resolve the frontend folder path — works regardless of where
# you run the script from
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend"
)

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# In-memory chat history: { session_id: [{"user": ..., "assistant": ...}] }
chat_sessions: dict = {}


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """
    Serve the frontend UI at http://localhost:5000
    Now you can open the browser at the root URL instead of
    double-clicking the HTML file.
    """
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check — returns server status."""
    return jsonify({
        "status":  "healthy",
        "model":   config.LLM_MODEL,
        "dataset": f"{len(documents)} records",
        "index":   config.PINECONE_INDEX_NAME,
    }), 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Request JSON:
    {
        "query":      "I have fever and cough",
        "session_id": "optional_user_id"
    }

    Response JSON:
    {
        "response":        "...",
        "confidence":      0.82,
        "confidence_pct":  82,
        "source":          "database",
        "used_web_search": false,
        "top_matches":     [...],
        "session_id":      "...",
        "processing_time": 1.23
    }
    """
    start_time = time.time()

    data       = request.get_json(silent=True) or {}
    query      = (data.get("query") or "").strip()
    session_id = data.get("session_id", "default")

    if not query:
        return jsonify({"error": "Missing 'query' field in request body"}), 400

    if len(query) > 1000:
        return jsonify({"error": "Query too long (max 1000 characters)"}), 400

    # Get or create session history
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    history = chat_sessions[session_id]

    try:
        result = agent.run(query=query, chat_history=history)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return jsonify({
            "error":   "Internal agent error. Please try again.",
            "details": str(e),
        }), 500

    # Update history (keep last 10 turns)
    history.append({"user": query, "assistant": result["response"]})
    chat_sessions[session_id] = history[-10:]

    processing_time = round(time.time() - start_time, 2)
    confidence_pct  = int(result["confidence"] * 100)

    logger.info(
        f"Chat done | session={session_id} | "
        f"conf={confidence_pct}% | source={result['source']} | {processing_time}s"
    )

    return jsonify({
        "response":        result["response"],
        "confidence":      result["confidence"],
        "confidence_pct":  confidence_pct,
        "source":          result["source"],
        "used_web_search": result["used_web_search"],
        "top_matches":     result["rag_results"],
        "session_id":      session_id,
        "processing_time": processing_time,
    }), 200


@app.route("/reset", methods=["POST"])
def reset_session():
    """Clear chat history for a session."""
    data       = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default")

    if session_id in chat_sessions:
        del chat_sessions[session_id]
        logger.info(f"Session reset: {session_id}")

    return jsonify({"message": f"Session '{session_id}' cleared"}), 200


@app.route("/sessions", methods=["GET"])
def list_sessions():
    """Debug — list active sessions and turn counts."""
    return jsonify({sid: len(turns) for sid, turns in chat_sessions.items()}), 200


# ─────────────────────────────────────────────────────────────
#  ERROR HANDLERS
# ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Cloud platforms (Railway, Render) inject PORT env var — always use it
    port = int(os.environ.get("PORT", config.FLASK_PORT))
    logger.info(f"Server starting on port {port}")
    logger.info(f"Frontend UI  → http://localhost:{port}/")
    logger.info(f"Health check → http://localhost:{port}/health")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=config.FLASK_DEBUG,
    )
