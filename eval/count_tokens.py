"""
Counts average input/output tokens across the 20 QA pairs
without calling the OpenAI API. Uses tiktoken to estimate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import tiktoken
from dotenv import load_dotenv
from src.retrieval.retrieval import retrieval
from src.utils.mongodb import mongo

load_dotenv()

QAS_PATH    = Path("eval/qas.json")
PROMPT_PATH = Path("prompts/rag_prompt.txt")
MODEL       = "gpt-4o-mini"
TOP_K       = 5


def build_context(results) -> str:
    chunks = []
    for i, r in enumerate(results, 1):
        title  = r.metadata.get("title", "Unknown")
        url    = r.metadata.get("url", "")
        source = f"[{i}] {title} — {url}" if url else f"[{i}] {title}"
        chunks.append(f"{source}\n{r.text}")
    return "\n\n---\n\n".join(chunks)


async def main():
    mongo.connect()

    qas          = json.loads(QAS_PATH.read_text(encoding="utf-8"))
    prompt_tpl   = PROMPT_PATH.read_text(encoding="utf-8")
    enc          = tiktoken.encoding_for_model(MODEL)

    input_counts  = []
    context_sizes = []

    for qa in qas:
        results = await retrieval.hybrid_rerank_search(
            query=qa["query"],
            top_k=TOP_K,
            metadata_filter={"source": qa["source"]},
        )
        context  = build_context(results)
        prompt   = prompt_tpl.format(context=context, question=qa["query"])
        n_input  = len(enc.encode(prompt)) + len(enc.encode(qa["query"]))
        input_counts.append(n_input)
        context_sizes.append(len(enc.encode(context)))
        print(f"  {qa['query'][:60]:<60} → {n_input} tokens")

    avg_input = sum(input_counts) / len(input_counts)
    avg_ctx   = sum(context_sizes) / len(context_sizes)

    print(f"\n{'─'*60}")
    print(f"Queries evaluated : {len(qas)}")
    print(f"Avg context tokens: {avg_ctx:.0f}")
    print(f"Avg input tokens  : {avg_input:.0f}")
    print(f"(Output tokens not estimated — measure from LangSmith)")

    mongo.close()


if __name__ == "__main__":
    asyncio.run(main())