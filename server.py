from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/predict")
async def predict():
    response = {
        "status": "success",
        "infer_time": 0.11,
        "predictions": [
            {
                "key": 1,
                "class": "dog",
                "confidence": 0.98
            },
        ]
    }
    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
