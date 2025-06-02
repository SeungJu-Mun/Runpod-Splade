from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-selfdistil")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-selfdistil")

@app.post("/v1/sparse-embeddings")
async def sparse_embedding(request: Request):
    data = await request.json()
    text = data["input"]
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[0].max(dim=0).values
        values, indices = torch.topk(scores, k=(scores > 0).sum())
        return {"embedding": {"indices": indices.tolist(), "values": values.tolist()}}