"""
it_agent.py — IT Support AI Agent with RAG (Local Gemma via Ollama)
====================================================================
Uses a local Gemma model (via Ollama) together with a VectorStore
to answer IT issues grounded in your documentation.

Setup:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve
    ollama pull gemma3
    pip install ollama --break-system-packages

Run:
    python it_agent.py
"""

import ollama
import os
from vector_store import VectorStore

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")


class ITSupportAgent:
    """
    Conversational IT support agent backed by RAG using a local Gemma model.

    Workflow per turn:
      1. Embed the user's issue and retrieve relevant doc chunks.
      2. Build a system prompt containing retrieved context.
      3. Send conversation history + new message to local Gemma via Ollama.
      4. Append Gemma's reply to history for multi-turn support.
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 5,
                 model: str = OLLAMA_MODEL):
        """
        Initialise the agent.

        Args:
            vector_store: Pre-built or pre-loaded VectorStore.
            top_k:        How many document chunks to retrieve per query.
            model:        Ollama model tag (default: env OLLAMA_MODEL or "gemma3").
        """
        self._check_ollama_available(model)

        self.vs = vector_store
        self.model = model
        self.conversation_history: list[dict] = []
        self.top_k = top_k
        print(f"✅  Using local model: {self.model}")

    # ── Ollama health check ──────────────────────────────────────

    @staticmethod
    def _check_ollama_available(model: str) -> None:
        """
        Verify Ollama is running and the requested model is pulled.
        Compatible with both old (dict) and new (object) Ollama SDK versions.
        """
        try:
            response = ollama.list()
            # New SDK: response.models is a list of objects with .model attribute
            # Old SDK: response["models"] is a list of dicts with "name" key
            if hasattr(response, "models"):
                available = [m.model for m in response.models]
            else:
                available = [m["name"] for m in response["models"]]
        except Exception as exc:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure it is running:\n"
                "    ollama serve\n"
                f"Original error: {exc}"
            ) from exc

        model_base = model.split(":")[0]
        matched = any(m.split(":")[0] == model_base for m in available)
        if not matched:
            raise RuntimeError(
                f"Model '{model}' not found in Ollama.\n"
                f"Pull it with:  ollama pull {model}\n"
                f"Available models: {available}"
            )

    # ── Retrieval ────────────────────────────────────────────────

    def retrieve_context(self, query: str) -> str:
        """Retrieve top-k relevant chunks and format as a context block."""
        chunks = self.vs.search(query, top_k=self.top_k)
        if not chunks:
            return "No relevant documentation found."

        sections = [f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)]
        return "\n\n---\n\n".join(sections)

    # ── System prompt ────────────────────────────────────────────

    def _build_system_prompt(self, context: str) -> str:
        return f"""You are an expert IT support engineer with deep knowledge of \
enterprise infrastructure, networking, and software systems.

Your job is to diagnose and resolve IT issues using ONLY the documentation \
excerpts provided below. Do NOT fabricate information that is not in the context.

=== DOCUMENTATION CONTEXT ===
{context}
=== END CONTEXT ===

Response guidelines:
• Provide clear, numbered step-by-step solutions.
• Reference excerpt numbers (e.g. "According to [Excerpt 2]…") when citing docs.
• If the documentation does not contain enough information, say so explicitly \
  and suggest escalation paths (e.g. vendor support, senior engineer review).
• Ask ONE clarifying question if the issue is ambiguous before proposing a fix.
• Summarise the root cause briefly before listing steps.
• End with a "Verification" section listing checks to confirm the fix worked."""

    # ── Main solve method ────────────────────────────────────────

    def solve(self, user_issue: str) -> str:
        """
        Process one user message and return the agent's response.

        Args:
            user_issue: Natural-language description of the IT problem.

        Returns:
            Agent's solution as a formatted string.
        """
        context = self.retrieve_context(user_issue)
        system_prompt = self._build_system_prompt(context)

        self.conversation_history.append({
            "role": "user",
            "content": user_issue,
        })

        messages_with_system = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
        ]

        response = ollama.chat(
            model=self.model,
            messages=messages_with_system,
            options={
                "temperature": 0.3,   # lower = more factual/deterministic
                "num_predict": 1024,  # max tokens to generate
                "top_p": 0.9,
            },
        )

        assistant_reply = response["message"]["content"]

        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_reply,
        })

        return assistant_reply

    # ── Utility ──────────────────────────────────────────────────

    def reset_conversation(self) -> None:
        """Clear conversation history to start a fresh session."""
        self.conversation_history = []
        print("🔄  Conversation history cleared.")


# ── Example usage ────────────────────────────────────────────────
if __name__ == "__main__":
    vs = VectorStore()
    vs.load("it_knowledge_base")

    agent = ITSupportAgent(vs, top_k=5)

    print(f"\n🤖  IT Support Agent ready (model: {agent.model}).")
    print("    Type 'exit' to quit, 'reset' to clear history.\n")

    while True:
        issue = input("🖥️  Your IT issue: ").strip()

        if issue.lower() in ("exit", "quit"):
            print("👋  Goodbye!")
            break
        elif issue.lower() == "reset":
            agent.reset_conversation()
            continue
        elif not issue:
            continue

        print("\n🤖  Agent:\n")
        solution = agent.solve(issue)
        print(solution)
        print("\n" + "─" * 60 + "\n")
