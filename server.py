from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict():
    response = {
        "status": "success",
        "infer_time": 0.5484125036746264,
        "predictions": [
            {
                "key": 0,
                "class": 0,
                "confidence": 0.94629967212677
            },
            {
                "key": 1,
                "class": 1,
                "confidence": 0.6092917323112488
            }
        ]
    }
    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
