import numpy
import requests
import tritonserver
from fastapi import FastAPI
from PIL import Image
from ray import serve


app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):
        self._triton_server = tritonserver

        model_repository = ["/mnt/models"]

        self._triton_server = tritonserver.Server(
            model_repository=model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )
        self._triton_server.start(wait_until_ready=True)

    @app.get("/generate")
    def generate(self, prompt: str, filename: str = "generated_image.jpg") -> None:
        if not self._triton_server.model("llama3-8b-instruct").ready():
            try:
                self._llama3_8b = self._triton_server.load("llama3-8b-instruct")
                if not self._llama3_8b.ready():
                    raise Exception("Model not ready")
            except Exception as error:
                print(f"Error can't load llama3_8b model, {error}")
                return
        responses = []
        for response in self._llama3_8b.infer(inputs={"prompt": [[prompt]]}):
            responses.append(response)

        return responses
