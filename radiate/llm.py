import os
import time
from radiate.metrics import log_llm_stats
from openai import OpenAI

class LLMClient:
    def __init__(
        self,
        provider: str = "openai",
        api_key: str = None,
        model: str = "gpt-3.5-turbo"
    ):
        self.provider = provider.lower()
        # Pick up API key from env if not passed
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        assert self.api_key, "Set your LLM API key (LLM_API_KEY or OPENAI_API_KEY)."
        self.model = model

        # Correct endpoint for OpenRouter
        self.api_base = "https://openrouter.ai/api/v1" if self.provider == "openrouter" else None

        # Instantiate client (new SDK style)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

    def answer(
        self,
        query: str,
        context_chunks,
        max_tokens: int = 512,
        system_prompt: str = None,
        model: str = None,
    ):
        prompt = self.format_prompt(query, context_chunks, system_prompt)
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": system_prompt or "You are a helpful assistant using retrieved context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
            )
            latency = time.time() - start

            # New SDK response structure
            answer_text = response.choices[0].message.content

            # Log metrics
            log_llm_stats(prompt, latency, len(answer_text.split()))

            tokens = getattr(response, "usage", {})
            return {
                "prompt": prompt,
                "answer": answer_text,
                "latency": latency,
                "tokens": tokens,
                "raw": response
            }
        except Exception as e:
            print(f"[LLM ERROR]: {e}")
            return {
                "error": str(e),
                "prompt": prompt,
                "answer": None
            }

    def format_prompt(self, query, context_chunks, system_prompt=None):
        # Accept context_chunks as list of dicts with 'text' OR just a string
        if isinstance(context_chunks, str):
            context_str = context_chunks
        elif isinstance(context_chunks, list):
            if len(context_chunks) > 0 and isinstance(context_chunks[0], dict) and "text" in context_chunks[0]:
                context_str = "\n\n".join([f"Chunk {i+1}: {c['text']}" for i, c in enumerate(context_chunks)])
            else:
                context_str = "\n\n".join([str(c) for c in context_chunks])
        else:
            context_str = str(context_chunks)
        return f"{context_str}\n\nQuestion: {query}"
