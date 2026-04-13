"""
Claude Code LLM backend — uses Max plan OAuth credentials from macOS keychain.

Routes through the standard Anthropic API using the OAuth token stored by
Claude Code in the macOS keychain. The billing header identifies the request
as coming from Claude Code, so usage is billed to the Max subscription.

This avoids:
- Paying for a separate API key
- The ~6s subprocess overhead of `claude -p` per call
- Any code changes to D-MEM's internals
"""

import json
import os
import re
import subprocess
import time
from typing import Optional

# Import D-MEM's interfaces
import sys
from pathlib import Path
DMEM_CODE = str(Path(__file__).parent / "vendor" / "dmem" / "code")
if DMEM_CODE not in sys.path:
    sys.path.insert(0, DMEM_CODE)

from memory_layer import BaseLLMController, TokenTracker, LLMController

token_tracker = TokenTracker()


def _get_oauth_token() -> str:
    """Read Claude Code OAuth token from macOS keychain."""
    result = subprocess.run(
        ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not read Claude Code credentials from keychain. "
            "Make sure you're logged into Claude Code."
        )
    creds = json.loads(result.stdout)
    return creds["claudeAiOauth"]["accessToken"]


def _extract_json(text: str) -> str:
    """Extract JSON from a response that may have markdown code fences or explanations."""
    # Try the raw text first
    text = text.strip()
    if text.startswith("{"):
        # Find the matching closing brace
        depth = 0
        for i, c in enumerate(text):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[: i + 1]

    # Try extracting from markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: find first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


class ClaudeCodeController(BaseLLMController):
    """LLM controller using Claude via Max plan OAuth credentials."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        from anthropic import Anthropic

        self.model = model
        self._token = _get_oauth_token()
        self.client = Anthropic(
            api_key=self._token,
            default_headers={
                "x-anthropic-billing-header": "cc_version=2.1.92.40a; cc_entrypoint=cli; cch=00000;"
            },
        )

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        """Get completion, compatible with D-MEM's OpenAIController interface."""
        from llm_cache import get_global_cache, LLMCache

        # Check cache first
        cache = get_global_cache()
        if cache is not None:
            key = LLMCache.make_key(prompt, self.model, temperature, response_format)
            cached = cache.get(key)
            if cached is not None:
                return cached

        # Build messages
        system_msg = "You must respond with a JSON object. No explanation, no markdown fences, just the raw JSON."
        if response_format and "json_schema" in response_format:
            schema = response_format["json_schema"].get("schema", {})
            system_msg += f"\n\nRequired JSON schema: {json.dumps(schema)}"

        t0 = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=temperature,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.time() - t0) * 1000

        # Track tokens
        token_tracker.record(
            response.usage.input_tokens,
            response.usage.output_tokens,
            latency_ms,
        )

        # Extract JSON from response
        raw_text = response.content[0].text
        result = _extract_json(raw_text)

        # Cache the result
        if cache is not None and result:
            cache.put(key, result, prompt, self.model)

        return result


def patch_llm_controller():
    """Monkey-patch LLMController to support 'claude-code' backend."""
    original_init = LLMController.__init__

    def patched_init(self, backend="sglang", model="gpt-4", api_key=None,
                     api_base=None, sglang_host="http://localhost", sglang_port=30000):
        if backend == "claude-code":
            self.llm = ClaudeCodeController(model=model)
        else:
            original_init(self, backend=backend, model=model, api_key=api_key,
                          api_base=api_base, sglang_host=sglang_host, sglang_port=sglang_port)

    LLMController.__init__ = patched_init


# Auto-patch on import
patch_llm_controller()
