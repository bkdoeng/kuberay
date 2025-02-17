import ray
from ray import serve
from fastapi import FastAPI
from tritonclient.grpc import service_pb2, service_pb2_grpc, model_config_pb2
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import grpc
import asyncio
from fastapi import FastAPI
from vllm import LLM, EngineArgs, SamplingParams
import os
import numpy as np
import json

app = FastAPI()

# Define the Triton server class
class TritonServer:
    def __init__(self):
        self.model_path = "/mnt/models"
        self.server = None
        self.port = 8081
        self.triton_process = None
        self.model_name = "llama3-8b-instruct"

    async def start(self):
        # Start Triton server as a subprocess
        command = [
            "/opt/tritonserver/bin/tritonserver",
            "--model-repository",
            self.model_path,
            "--grpc-port",
            str(self.port),
            "--log-info",
            bool(true),
            "--log-error",
            bool(true),
            "--log-warning",
            bool(true),
        ]
        self.triton_process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Wait for Triton server to start
        await asyncio.sleep(5)

    async def stop(self):
         if self.triton_process:
            self.triton_process.terminate()
            await self.triton_process.wait()

    async def infer(self, prompt, sampling_params):
        try:
            # Create a gRPC client
            options = [('grpc.max_send_message_length', 100 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
            async with grpc.aio.insecure_channel(f"localhost:{self.port}", options=options) as channel:
                stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
                
                # Create request
                request = service_pb2.ModelInferRequest()
                request.model_name = self.model_name
                request.model_version = "4294967296"

                # Prepare input tensors
                input_data = np.array([prompt]).astype(np.object_)
                #input_tensor = grpcclient.InferInput("text_input", input_data.shape, "BYTES")
                #input_tensor.set_data_from_numpy(input_data)
                
                input_proto = service_pb2.ModelInferRequest().InferInputTensor()
                input_proto.name = "text_input"
                input_proto.shape.extend(input_data.shape)
                input_proto.datatype = "BYTES"
                input_proto.contents.raw_contents.append(input_tensor.serialize())
                request.inputs.extend([input_proto])
                
                #Prepare sampling parameters
                sampling_params_str = json.dumps(sampling_params)
                params_data = np.array([sampling_params_str]).astype(np.object_)
                params_tensor = grpcclient.InferInput("sampling_params", params_data.shape, "BYTES")
                params_tensor.set_data_from_numpy(params_data)

                params_proto = service_pb2.ModelInferRequest().InferInputTensor()
                params_proto.name = "sampling_params"
                params_proto.shape.extend(params_data.shape)
                params_proto.datatype = "BYTES"
                params_proto.contents.raw_contents.append(params_tensor.serialize())
                request.inputs.extend([params_tensor])
                
                # Prepare output tensors
                output_tensor = grpcclient.InferRequestedOutput("text_output")
                request.outputs.extend([output_tensor])

                # Send request and get response
                response = await stub.ModelInfer(request)

                # Process response
                output = grpcclient.InferResult(response)
                generated_text = output.as_numpy("text_output")[0].decode("utf-8")

                return generated_text

        except grpc.aio.AioRpcError as e:
            print(f"gRPC error: {e}")
            return None

# Define the Ray Serve deployment
@serve.deployment()
@serve.ingress(app)
class InferService:
    def __init__(self):
        self.triton_server = TritonServer()
        self.triton_server.start()
        self.model_path = "/mnt/models"

    @app.get("/generate_stream")
    def stream_inference(server_url="localhost:8081", model_name="llama3-8b-instruct", input_text="history of chennai", sequence_id=1234, stream_timeout=30):
        try:
            client = grpcclient.InferenceServerClient(url=server_url)
    
            # Define the input
            input_data = np.array([input_text], dtype=np.object_)
            input_tensor = grpcclient.InferInput("text_input", input_data.shape, "BYTES")
            input_tensor.set_data_from_numpy(input_data)
    
            # Define output
            output_tensor = grpcclient.InferRequestedOutput("text_output")
    
            # Define callback function
            def callback(result, error):
                if error:
                    print(f"Inference error: {error}")
                else:
                    response = result.get_response()
                    output_data = result.as_numpy("text_output")
                    print(f"Received: {output_data[0].decode('utf-8')}")
    
            # Start streaming
            client.start_stream(callback=callback, stream_timeout=stream_timeout)
    
            # Send request
            client.async_stream_infer(
                model_name=model_name,
                inputs=[input_tensor],
                request_id=str(sequence_id),
                outputs=[output_tensor],
                sequence_id=sequence_id,
                sequence_start=True,
                sequence_end=True
            )
    
            # Wait for stream to complete (adjust as needed)
            import time
            time.sleep(stream_timeout + 1)
    
            client.stop_stream()
            return "success"
    
        except grpcclient.InferenceServerException as e:
            print(f"Error during inference: {e}")
    
    @app.get("/test")
    async def test(self):
        prompt = "what is triton"
        sampling_params = '{"temperature":{"double_param":0.7},"top_p":{"double_param":0.9},"max_tokens":{"int64_param":256}}'

        if not prompt:
            return "Error: Prompt is required."
        
        result = await self.triton_server.infer(prompt, sampling_params)
        return {"result": result}

app = InferService.bind()
