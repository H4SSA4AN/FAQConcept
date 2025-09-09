import json
from pathlib import Path
from loguru import logger

from app.retrieval import answer


def main():
    samples_path = Path(__file__).parent / "sample_queries.json"
    data = json.loads(samples_path.read_text(encoding="utf-8"))

    all_ok = True
    for item in data:
        q = item["query"]
        logger.info(f"Query: {q}")
        res = answer(q, k=10)
        if res.get("mode") == "answer":
            logger.info(f"ANSWER: {res['question']} | score={res['score']:.3f}")
        else:
            logger.info("SUGGEST:")
            for c in res["candidates"]:
                logger.info(f"  - {c['question']} | score={c['score']:.3f}")

    logger.info("Smoke test complete.")


if __name__ == "__main__":
    main()


