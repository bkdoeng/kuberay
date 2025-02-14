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
            log_info=True,
            log_error=True,
            log_warn=True,
            backend_directory='/opt/tritonserver/backends',
            metrics=True, gpu_metrics=True, cpu_metrics=True,
        )
        self._triton_server.start(wait_until_ready=True)
        self._llama3_8b = self._triton_server.load("llama3-8b-instruct")
        if not self._llama3_8b.ready():
                    raise Exception("Model not ready")
            
    @app.get("/infer")        
    def infer(self, server_url="http://localhost:8001", model_name="llama3-8b-instruct", prompt="what is tritonserver", max_tokens=1000, temperature=0.7):
        print(f"Server Live: {self._triton_server.live()}")
        print(f"Server Ready: {self._triton_server.ready()}")
        print(f"Server Metadata: {self._triton_server.metadata()}")
        
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

        print("Inferencing")
        response = self._llama3_8b.infer(
            inputs={"text_input":["what is tritonserver?"]},
        )
        print("Inference done")
        
        # Extract and return the generated text
        #result = response[0]
        print(f"Here is the result: \n{response}")
        output_data = response.as_numpy("text_output")
        print(f"Here is the output: \n{output_data}")
        #generated_text = output_data[0].decode()
        return "success"

    @app.post("/httptest")
    def httptest(self):
        # Triton server details
        triton_url = "localhost:8082"
        model_name = "llama3-8b-instruct"
        input_name = "text_input"
        output_name = "text_output"
        
        # Create Triton HTTP client
        try:
            triton_client = grpcclient.InferenceServerClient(url=triton_url)
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
        #input_text = "What is the capital of France?"
        #inputs = [httpclient.InferInput(input_name, [1], "BYTES")]
        #inputs[0].set_data_from_numpy(np.array([input_text.encode('utf-8')], dtype=np.object_))
        
        # Output configuration
        #outputs = [httpclient.InferRequestedOutput(output_name)]
        
        # Perform inference
        #try:
        #    response = triton_client.infer(
        #        model_name=model_name,
        #        inputs=inputs,
        #        outputs=outputs
        #    )
        #except Exception as e:
        #    print(f"inference error: {e}")
        #    exit(1)
        
        # Process output
        #result = response.get_response()
        #output_data = response.as_numpy(output_name)
        #generated_text = output_data[0].decode('utf-8')
        #print(f"Generated text: {generated_text}")
        return "success" 
        
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
