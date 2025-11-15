"""
LLM client for generating answers from retrieved context.
Supports OpenAI, OpenRouter, and compatible providers.
"""

import os
import time
from typing import Dict, Any, List, Union, Optional
from radiate.metrics import log_llm_stats
from openai import OpenAI


class LLMClient:
    """
    Universal LLM client supporting multiple providers.
    
    Supports:
        - OpenAI (GPT-3.5, GPT-4, etc.)
        - OpenRouter (access to multiple models)
        - Any OpenAI-compatible API
    
    Attributes:
        provider: LLM provider name
        api_key: API key for authentication
        model: Model identifier
        client: OpenAI client instance
    """
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: Provider name ("openai", "openrouter", etc.)
            api_key: API key (falls back to env vars LLM_API_KEY or OPENAI_API_KEY)
            model: Model identifier
            
        Raises:
            AssertionError: If no API key is provided or found in environment
        """
        self.provider = provider.lower()
        
        # Pick up API key from env if not passed
        self.api_key = (
            api_key or 
            os.environ.get("LLM_API_KEY") or 
            os.environ.get("OPENAI_API_KEY")
        )
        
        assert self.api_key, (
            "No API key provided. Set LLM_API_KEY or OPENAI_API_KEY environment variable, "
            "or pass api_key parameter."
        )
        
        self.model = model
        
        # Set API base URL for different providers
        if self.provider == "openrouter":
            self.api_base = "https://openrouter.ai/api/v1"
        else:
            self.api_base = None  # Use default OpenAI endpoint
        
        # Instantiate OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def answer(
        self,
        query: str,
        context_chunks: Union[str, List[Dict[str, str]], List[str]],
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate an answer using retrieved context.
        
        Args:
            query: User question
            context_chunks: Retrieved context (string, list of dicts, or list of strings)
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt (optional)
            model: Override default model (optional)
            temperature: Sampling temperature (0-2, default 0.7)
            
        Returns:
            Dict containing:
                - prompt: Formatted prompt sent to LLM
                - answer: Generated answer text
                - latency: Response time in seconds
                - tokens: Token usage statistics
                - raw: Raw API response
                
        Example:
            >>> llm = LLMClient(provider="openai", api_key="sk-...")
            >>> chunks = radiate.query("what is ML?", rerank=True)
            >>> response = llm.answer("what is ML?", chunks)
            >>> print(response['answer'])
        """
        # Format context into prompt
        prompt = self.format_prompt(query, context_chunks, system_prompt)
        
        # Default system prompt
        default_system_prompt = (
            "You are a helpful assistant. Answer the user's question based on "
            "the provided context. If the context doesn't contain enough information, "
            "say so clearly."
        )
        
        start = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt or default_system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency = time.time() - start
            
            # Extract answer from response
            answer_text = response.choices[0].message.content
            
            # Extract token usage
            usage = response.usage if hasattr(response, 'usage') else None
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            # Log metrics (FIXED)
            log_llm_stats(
                provider=self.provider,
                model=model or self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency,
                cost=0.0  # TODO: Calculate actual cost based on provider pricing
            )
            
            return {
                "prompt": prompt,
                "answer": answer_text,
                "latency": latency,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "raw": response
            }
            
        except Exception as e:
            print(f"[LLM ERROR]: {e}")
            return {
                "error": str(e),
                "prompt": prompt,
                "answer": None,
                "latency": time.time() - start
            }
    
    def format_prompt(
        self,
        query: str,
        context_chunks: Union[str, List[Dict[str, str]], List[str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format query and context into a prompt.
        
        Args:
            query: User question
            context_chunks: Context (string, list of dicts, or list of strings)
            system_prompt: Optional system prompt (not used in this method, kept for compatibility)
            
        Returns:
            Formatted prompt string
        """
        # Handle different context formats
        if isinstance(context_chunks, str):
            context_str = context_chunks
        elif isinstance(context_chunks, list):
            if len(context_chunks) > 0 and isinstance(context_chunks[0], dict):
                # List of dicts with 'text' field
                context_parts = []
                for i, chunk in enumerate(context_chunks):
                    text = chunk.get('text', str(chunk))
                    context_parts.append(f"[Context {i+1}]\n{text}")
                context_str = "\n\n".join(context_parts)
            else:
                # List of strings
                context_str = "\n\n".join([str(c) for c in context_chunks])
        else:
            context_str = str(context_chunks)
        
        # Build final prompt
        prompt = (
            f"Context:\n"
            f"{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer based on the context above:"
        )
        
        return prompt
