import requests
import tritonserver
from fastapi import FastAPI
from PIL import Image
from ray import serve
import grpc
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
import tritonclient.http as httpclient
import json


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
        self._llama3_8b = self._triton_server.load("llama3-8b-instruct")
        if not self._llama3_8b.ready():
                    raise Exception("Model not ready")
            
    @app.get("/infer")        
    def infer(server_url="http://localhost:8001", model_name="llama3-8b-instruct", prompt="what is tritonserver", max_tokens=1000, temperature=0.7):
        try:
            client = grpcclient.InferenceServerClient(url=server_url)
        except Exception as e:
            print(f"Failed to connect to Triton server: {e}")
            return None

        health_ctx = client.is_server_live()
        print(f"Server live: {health_ctx}")
        
        ready_ctx = client.is_server_ready()
        print(f"Server ready: {ready_ctx}")
        
        model_ready_ctx = client.is_model_ready(model_name="llama3-8b-instruct", model_version="4294967296")
        print(f"Model ready: {model_ready_ctx}")
        
        server_metadata = client.get_server_metadata()
        print(f"Server metadata: {server_metadata}")
        
        model_metadata = client.get_model_metadata(model_name="llama3-8b-instruct", model_version="4294967296")
        print(f"Model metadata: {model_metadata}")

        #print("Tritonserver connection is successful")
        # Define input and output tensors
        input_data = np.array([prompt], dtype=np.object_)
        input_tensor = grpcclient.InferInput("text_input", input_data.shape, "BYTES")
        input_tensor.set_data_from_numpy(input_data)
    
        # Set inference parameters
        parameters = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "bad_words": "",
            "stop_words": ""
        }
    
        # Create request
        inputs = [input_tensor]
        outputs = [grpcclient.InferRequestedOutput("text_output")]
    
        try:
            print("Inferencing")
            response = self._llama3_8b.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
                parameters=parameters
            )
            print("Inference done")
            
            # Extract and return the generated text
            result = response.get_response()
            print(f"Here is the result: \n{result}")
            output_data = response.as_numpy("text_output")
            generated_text = output_data[0].decode()
            return generated_text
    
        except grpcclient.RequestError as e:
            print(f"Inference failed: {e}")
            return None

    @app.post("/httptest")
    def httptest(self):
        # Triton server details
        triton_url = "localhost:8000"
        model_name = "llama3-8b-instruct"
        input_name = "text_input"
        output_name = "text_output"
        
        # Create Triton HTTP client
        try:
            triton_client = httpclient.InferenceServerClient(url=triton_url)
        except Exception as e:
            print(f"channel creation error: {e}")
            exit(1)
        
        # Check server and model status
        if not triton_client.is_server_live():
            print("server is not live")
            exit(1)
        
        if not triton_client.is_model_ready(model_name):
            print(f"{model_name} is not ready")
            exit(1)
            
        # Input data
        input_text = "What is the capital of France?"
        inputs = [httpclient.InferInput(input_name, [1], "BYTES")]
        inputs[0].set_data_from_numpy(np.array([input_text.encode('utf-8')], dtype=np.object_))
        
        # Output configuration
        outputs = [httpclient.InferRequestedOutput(output_name)]
        
        # Perform inference
        try:
            response = triton_client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
        except Exception as e:
            print(f"inference error: {e}")
            exit(1)
        
        # Process output
        result = response.get_response()
        output_data = response.as_numpy(output_name)
        generated_text = output_data[0].decode('utf-8')
        print(f"Generated text: {generated_text}")
        return generated_text 
        
    @app.get("/generate")
    def generate(self, prompt: str) -> None:
        #if not self._triton_server.model("llama3-8b-instruct").ready():
        #    try:
        #        self._llama3_8b = self._triton_server.load("llama3-8b-instruct")
        #        if not self._llama3_8b.ready():
        #            raise Exception("Model not ready")
        #    except Exception as error:
        #        print(f"Error can't load llama3_8b model, {error}")
        #        return
        #responses = []
        #for response in self._llama3_8b.infer(inputs={"prompt": [[prompt]]}):
        #    responses.append(response)

        return "test success"

triton = TritonDeployment.bind()
