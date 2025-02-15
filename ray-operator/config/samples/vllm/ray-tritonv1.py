import ray
from ray import serve
from tritonclient.grpc import service_pb2, service_pb2_grpc, model_config_pb2
import tritonclient.grpc as grpcclient
import asyncio
from fastapi import FastAPI
import grpc
app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class TritonServer:
    def __init__(self):
        self.model_repository_path = "/mnt/models"
        self.model_name = "llama3-8b-instruct"
        self.triton_server = self._start_triton_server()
        self.channel = grpc.insecure_channel("localhost:8081")
        self.stub = service_pb2_grpc.GRPCInferenceServiceStub(self.channel)
        self._load_model()

    async def __aenter__(self):
        self.triton_server = await self._start_triton_server()
        self.channel = grpc.insecure_channel("localhost:8081")
        self.stub = service_pb2_grpc.GRPCInferenceServiceStub(self.channel)
        await self._load_model()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._unload_model()
        await self._stop_triton_server()
        if self.channel:
            await self.channel.close()

    async def _start_triton_server(self):
      process = await asyncio.create_subprocess_exec(
          "/opt/tritonserver/bin/tritonserver",
          "--model-repository",
          self.model_repository_path,
          "--grpc-port",
          "8081",
          "--log-info",
          "true",
          "--log-error",
          "true",
          "--log-warning",
          "true",
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.PIPE
      )
      return process

    async def _stop_triton_server(self):
        if self.triton_server:
            self.triton_server.terminate()
            await self.triton_server.wait()

    async def _load_model(self):
        request = service_pb2.RepositoryModelLoadRequest(model_name=self.model_name)
        await self.stub.RepositoryModelLoad(request)

    async def _unload_model(self):
        request = service_pb2.RepositoryModelUnloadRequest(model_name=self.model_name)
        await self.stub.RepositoryModelUnload(request)

    @app.post("/infer")
    async def test(self):
        input_data = {
            "inputs": {
                "text_input": {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["What is triton"],
                }
            },
            "outputs": ["text_output"],
        }
        async with self:
          result = await self.infer(input_data.json())
        return result
        
    async def infer(self, request: dict) -> dict:
        inputs = []
        outputs = []
        for input_name, input_data in request["inputs"].items():
            triton_input = grpcclient.InferInput(input_name, input_data["shape"], input_data["datatype"])
            triton_input.set_data_from_numpy(np.array(input_data["data"], dtype=input_data["datatype"]))
            inputs.append(triton_input)
        for output_name in request["outputs"]:
            outputs.append(grpcclient.InferRequestedOutput(output_name))
        infer_request = service_pb2.ModelInferRequest()
        infer_request.model_name = self.model_name
        infer_request.inputs.extend(inputs)
        infer_request.outputs.extend(outputs)
        response = await self.stub.ModelInfer(infer_request)
        result = {}
        for output in response.outputs:
            result[output.name] = grpcclient.InferResult(response).as_numpy(output.name).tolist()
        return result
    
app = TritonServer.bind()
