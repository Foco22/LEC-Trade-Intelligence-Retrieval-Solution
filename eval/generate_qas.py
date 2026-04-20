"""
Genera 20 pares query/answer desde chunks reales de MongoDB:
  - 7 desde FSA
  - 7 desde GOV.UK
  - 6 desde WTO

Uso:
    python eval/generate_qas.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import random
from dotenv import load_dotenv
from src.utils.mongodb import mongo
from src.utils.llm import llm

load_dotenv()

OUTPUT_PATH = Path("eval/qas.json")

SAMPLE_CONFIG = {
    "fsa":   7,
    "govuk": 7,
    "wto":   6,
}

SYSTEM_PROMPT = """You are a question generation assistant.
Given a text chunk from a regulatory document, generate one clear question
that can be answered using ONLY the information in that chunk, and provide
the answer based strictly on the chunk.

Respond in JSON with exactly these fields:
{
  "query": "the question",
  "answer": "the answer based on the chunk"
}
Only return valid JSON, nothing else."""


async def sample_chunks(source: str, n: int) -> list[dict]:
    """Saca n chunks aleatorios de una fuente."""
    col = mongo.get_collection()
    pipeline = [
        {"$match": {"metadata.source": source}},
        {"$sample": {"size": n * 3}},  # extra por si algún chunk es muy corto
        {"$project": {"embedding": 0}},
    ]
    docs = await col.aggregate(pipeline).to_list(n * 3)
    # Filtrar chunks muy cortos
    docs = [d for d in docs if len(d.get("text", "")) > 300]
    return docs[:n]


def generate_qa(chunk: dict) -> dict | None:
    """Llama al LLM para generar un par query/answer desde un chunk."""
    user_prompt = f"Generate a question and answer from this chunk:\n\n{chunk['text'][:1500]}"
    try:
        response = llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        data = json.loads(response.content)
        return {
            "query":           data["query"],
            "answer":          data["answer"],
            "source":          chunk["metadata"]["source"],
            "doc_id":          chunk["doc_id"],
            "relevant_topics": [chunk["metadata"].get("topic", "")],
        }
    except Exception as e:
        print(f"  Error generando QA para {chunk['doc_id']}: {e}")
        return None


async def main():
    mongo.connect()
    qas = []

    for source, n in SAMPLE_CONFIG.items():
        print(f"\nGenerando {n} QAs desde {source}...")
        chunks = await sample_chunks(source, n)

        for chunk in chunks:
            qa = generate_qa(chunk)
            if qa:
                qas.append(qa)
                print(f"  ✓ [{qa['source']}] {qa['query'][:70]}...")

    random.shuffle(qas)
    OUTPUT_PATH.write_text(json.dumps(qas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(qas)} QAs guardados en {OUTPUT_PATH}")
    mongo.close()


if __name__ == "__main__":
    asyncio.run(main())