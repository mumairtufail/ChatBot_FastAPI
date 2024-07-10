from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from huggingface_hub import InferenceClient

# Initialize FastAPI app
app = FastAPI()

# Initialize InferenceClient
client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token="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",
)

@app.post("/chat")
async def chat(message: str = Form(...)):
    try:
        response = ""
        for msg in client.chat_completion(
            messages=[{"role": "user", "content": message}],
            max_tokens=500,
            stream=True,
        ):
            response += msg.choices[0].delta.content
        return {
            "code": 200,
            "status": "success",
            "response": response
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))