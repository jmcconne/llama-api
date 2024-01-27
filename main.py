from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    model: str
    temp: float = 0.0
    streaming: bool = True

@app.post("/complete")
def perform_inference(request: InferenceRequest):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    if request.model == "llama-2-7b":
        model_path = "models/llama-2-7b/ggml-model-q8_0.gguf"
    elif request.model == "llama-2-13b":
        model_path = "models/llama-2-13b/ggml-model-q8_0.gguf"
    llm = LlamaCpp(
        model_path=model_path,
        temperature=request.temp,
        n_ctx=3072,
        n_gpu_layers=1,  # Metal set to 1 is enough.
        n_batch=512, # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True, # Verbose is required to pass to the callback manager
        max_tokens=1024,
        streaming=request.streaming)
    #response = llm.stream(request.prompt)
    if request.streaming == True:
        response = llm.stream(request.prompt)
        return StreamingResponse(response)
    else:
        response = llm.complete(request.prompt)
        return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
