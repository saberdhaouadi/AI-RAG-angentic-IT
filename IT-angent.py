"""
it_agent.py — Step 3: IT Support AI Agent with RAG
===================================================
Uses Claude (via the Anthropic API) together with a VectorStore
to answer IT issues grounded in your documentation.

Install:
    pip install anthropic --break-system-packages
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import anthropic
import os
from vector_store import VectorStore


class ITSupportAgent:
    """
    Conversational IT support agent backed by RAG.

    Workflow per turn:
      1. Embed the user's issue and retrieve relevant doc chunks.
      2. Build a system prompt containing retrieved context.
      3. Send conversation history + new message to Claude.
      4. Append Claude's reply to history for multi-turn support.

    Attributes:
        vs:                   VectorStore instance for retrieval.
        client:               Anthropic API client.
        conversation_history: Full multi-turn conversation list.
        top_k:                Number of chunks to retrieve per query.
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        """
        Initialise the agent.

        Args:
            vector_store: Pre-built or pre-loaded VectorStore.
            top_k:        How many document chunks to retrieve per query.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set.\n"
                "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
            )

        self.vs = vector_store
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history: list[dict] = []
        self.top_k = top_k

    # ── Retrieval ────────────────────────────────────────────────

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve top-k relevant chunks and format as a context block.

        Args:
            query: The user's IT issue description.

        Returns:
            Formatted string of retrieved documentation excerpts.
        """
        chunks = self.vs.search(query, top_k=self.top_k)
        if not chunks:
            return "No relevant documentation found."

        # Number each excerpt so the model can reference them
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
        # Step 1: Retrieve relevant documentation
        context = self.retrieve_context(user_issue)

        # Step 2: Build dynamic system prompt with retrieved context
        system_prompt = self._build_system_prompt(context)

        # Step 3: Append user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_issue,
        })

        # Step 4: Call Claude with full conversation history
        response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=system_prompt,
            messages=self.conversation_history,
        )

        assistant_reply = response.content[0].text

        # Step 5: Store reply for multi-turn continuity
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
    # Load pre-built knowledge base
    vs = VectorStore()
    vs.load("it_knowledge_base")

    agent = ITSupportAgent(vs, top_k=5)

    print("🤖  IT Support Agent ready. Type 'exit' to quit, 'reset' to clear history.\n")

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

