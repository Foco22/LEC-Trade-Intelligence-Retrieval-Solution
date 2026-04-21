from pathlib import Path
from langsmith import traceable
from src.retrieval.retrieval import SearchResult
from src.utils.llm import LLM, LLMResponse

PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "rag_prompt.txt"


class Generate:
    def __init__(self):
        self._llm    = LLM()
        self._prompt = PROMPT_PATH.read_text(encoding="utf-8")

    @traceable(name="clara-answer")
    def answer(self, query: str, results: list[SearchResult]) -> LLMResponse:
        context = self._build_context(results)
        prompt  = self._prompt.format(context=context, question=query)
        return self._llm.generate(system_prompt=prompt, user_prompt=query)

    def _build_context(self, results: list[SearchResult]) -> str:
        chunks = []
        for i, r in enumerate(results, 1):
            title  = r.metadata.get("title", "Unknown")
            url    = r.metadata.get("url", "")
            source = f"[{i}] {title} — {url}" if url else f"[{i}] {title}"
            chunks.append(f"{source}\n{r.text}")
        return "\n\n---\n\n".join(chunks)


generate = Generate()