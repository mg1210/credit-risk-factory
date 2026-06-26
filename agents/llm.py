"""
core/llm.py
───────────
Thin wrapper around the Anthropic API.
Agents call ask() to get LLM-generated reasoning, narratives, and recommendations.
"""

import os
import json
import anthropic


_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = ""

        # 1) Environment variable
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

        # 2) Local api_key.txt in project root
        if not api_key:
            key_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api_key.txt")
            if os.path.exists(key_file):
                with open(key_file, encoding="utf-8") as f:
                    api_key = f.read().strip()

        # 3) Streamlit secrets (Streamlit Cloud)
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
            except Exception:
                pass

        if not api_key:
            raise RuntimeError(
                "No Anthropic API key found. Provide it via:\n"
                "  1) ANTHROPIC_API_KEY environment variable\n"
                "  2) api_key.txt file in the project root\n"
                "  3) Streamlit secrets (ANTHROPIC_API_KEY)"
            )

        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def ask(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    """Send a prompt to Claude and return the text response. Returns '' if no API key."""
    try:
        client = _get_client()
    except RuntimeError as e:
        print(f"  LLM skipped — {e.args[0].splitlines()[0]}")
        return ""
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=messages,
    )
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text


def ask_json(prompt: str, system: str = "", max_tokens: int = 1024) -> dict:
    """Ask Claude for a JSON response. Strips markdown fences before parsing."""
    raw = ask(prompt, system=system, max_tokens=max_tokens)
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
        clean = clean.rsplit("```", 1)[0]
    return json.loads(clean.strip())


CREDIT_RISK_SYSTEM = """
You are an expert credit risk data scientist and model validator with 15+ years of
experience in consumer lending at major banks and NBFCs. You understand regulatory
requirements, model governance, IFRS 9, Basel frameworks, and best practices in
scorecard development. You communicate clearly to both technical and business audiences.
Always be precise, concise, and practical.
"""
