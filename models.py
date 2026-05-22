from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Iterable, List, Optional

from ollama import Client
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.ollama_llm import OllamaLLM


_original_invoke = OllamaLLM.invoke


def _patched_invoke(self, input, message_history=None, system_instruction=None):
    if message_history is not None and hasattr(message_history, "messages"):
        message_history = message_history.messages

    def _call_ollama():
        return self.client.chat(
            model=self.model_name,
            messages=self.get_messages(input, message_history, system_instruction),
            **self.model_params,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_ollama)
            response = future.result(timeout=120)
    except FutureTimeoutError:
        print("Ollama call timed out after 120 seconds")
        return LLMResponse(content="[LLM response timeout]")
    except Exception as exc:
        print(f"Ollama call failed: {str(exc)[:100]}")
        return LLMResponse(content="[LLM response error]")

    if isinstance(response, dict):
        content = response.get("message", {}).get("content", "")
    elif hasattr(response, "message"):
        content = response.message.content or ""
    else:
        content = str(response)

    return LLMResponse(content=content)


OllamaLLM.invoke = _patched_invoke

print("Patched OllamaLLM.invoke for dict-style Ollama responses")


class OllamaVectorEmbedder:
    def __init__(self, client: Client, model: str, max_length: int = 8000):
        self._client = client
        self._model = model
        self._dimension: Optional[int] = None
        self._max_length = max_length

    def embed_query(self, text: str) -> List[float]:
        if len(text) > self._max_length:
            print(f"Embedding input truncated from {len(text)} to {self._max_length} characters")
            text = text[: self._max_length]

        try:
            resp = self._client.embeddings(model=self._model, prompt=text or " ")
            return resp["embedding"]
        except Exception as exc:
            if "context length" in str(exc).lower() or "input length exceeds" in str(exc).lower():
                shorter = max(1, self._max_length // 2)
                print(f"Embedding retry with shorter input: {shorter} characters")
                text = text[:shorter]
                resp = self._client.embeddings(model=self._model, prompt=text or " ")
                return resp["embedding"]
            raise

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension
