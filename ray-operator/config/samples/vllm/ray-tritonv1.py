import ray
from ray import serve
from tritonclient.grpc import service_pb2, service_pb2_grpc, model_config_pb2
import tritonclient.grpc as grpcclient
import asyncio

@serve.deployment(
    name="TritonServer"
)
class TritonServer:
    def __init__(self, model_repository_path: str, model_name: str):
        self.model_repository_path = model_repository_path
        self.model_name = model_name
        self.triton_server = None
        self.channel = None
        self.stub = None

    async def __aenter__(self):
        self.triton_server = await self._start_triton_server()
        self.channel = grpcclient.insecure_channel("localhost:8001")
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
          "tritonserver",
          "--model-repository",
          self.model_repository_path,
          "--grpc-port",
          "8001",
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.PIPE
      )
      return process

    async def _stop_triton_server(self):
        if self.triton_server:
            self.triton_server.terminate()
            await self.triton_server.wait()

    async def _load_model(self):
        request = service_pb2.ModelLoadRequest(model_name=self.model_name)
        await self.stub.ModelLoad(request)

    async def _unload_model(self):
        request = service_pb2.ModelUnloadRequest(model_name=self.model_name)
        await self.stub.ModelUnload(request)

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

    async def __call__(self, http_request):
      request_data = await http_request.json()
      async with self:
        result = await self.infer(request_data)
      return result

@serve.deployment
class VLLMService:
    def __init__(self, triton_server: TritonServer):
        self.triton_server = triton_server

    async def __call__(self, request: str) -> str:
        input_data = {
            "inputs": {
                "text_input": {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [request],
                }
            },
            "outputs": ["text_output"],
        }
        result = await self.triton_server.__call__(input_data)
        return result["text_output"][0]
    
app = VLLMService.bind(TritonServer.bind(model_repository_path="/mnt/model", model_name="llama3-8b-instruct"))
