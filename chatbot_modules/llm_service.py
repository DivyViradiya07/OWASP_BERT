import asyncio
import time
from typing import Dict, Any, Tuple

from chatbot_modules.config import CONFIG

llm_response_cache: Dict[Tuple, Tuple[str, float]] = {}

async def generate_llm_response(llm_model, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate a response using the LLM with error handling, retries, and caching."""
    cache_key = (prompt, max_tokens, temperature)

    if cache_key in llm_response_cache:
        response, timestamp = llm_response_cache[cache_key]
        if time.time() - timestamp < CONFIG["LLM_RESPONSE_CACHE_TTL_SEC"]:
            return response

    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            response = await asyncio.to_thread(
                llm_model.create_completion,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=CONFIG["TOP_P"],
                echo=False
            )

            if response and 'choices' in response and response['choices']:
                generated_text = response['choices'][0]['text'].strip()
                llm_response_cache[cache_key] = (generated_text, time.time())
                return generated_text
            raise ValueError("Empty or invalid response from LLM")

        except Exception as e:
            print(f"Error generating LLM response (attempt {attempt + 1}/{CONFIG['MAX_RETRIES']}): {e}")
            if attempt < CONFIG["MAX_RETRIES"] - 1:
                sleep_time = CONFIG["INITIAL_BACKOFF_SEC"] * (2 ** attempt)
                print(f"Retrying in {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)

    return "I'm sorry, I encountered an error while processing your request. Please try again later."