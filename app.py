from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, root_validator
from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "BAAI/bge-reranker-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model {model_name} loaded successfully on {device}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

class RerankRequest(BaseModel):
    query: str
    results: List[dict]

    @root_validator(pre=True)
    def normalize_results(cls, values):
        results = values.get("results", [])
        normalized_results = []
        for result in results:
            if not isinstance(result, dict):
                logger.warning(f"Skipping invalid result: {result}")
                continue
            content = (
                result.get("content") or
                result.get("text") or
                result.get("description") or
                result.get("rule_text") or
                result.get("body") or
                ""
            )
            normalized_results.append({"content": content, **result})
        values["results"] = normalized_results
        return values

class RerankResponse(BaseModel):
    ranked_results: List[dict]

@app.post("/rerank", response_model=RerankResponse)
async def rerank_results(request: RerankRequest):
    try:
        query = request.query
        results = request.results
        print(query)

        input_pairs = [[query, result.get("content", "")] for result in results]

        inputs = tokenizer(
            input_pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            scores = model(**inputs).logits.squeeze(-1).cpu().numpy()

        ranked_results = [
            {**result, "relevanceScore": float(score)}
            for result, score in zip(results, scores)
        ]

        ranked_results = sorted(
            ranked_results,
            key=lambda x: x["relevanceScore"],
            reverse=True
        )[:50]

        return RerankResponse(ranked_results=ranked_results)

    except Exception as e:
        logger.error(f"Re-ranking error: {str(e)}")
        return RerankResponse(ranked_results=results[:50])

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)