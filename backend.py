from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from ollama import Client
import json
import logging
from typing import Dict
from fastapi.responses import PlainTextResponse
import uuid
import requests
from PIL import Image
from io import BytesIO
import base64
import tempfile
import os
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# In-memory storage for simplicity
storage: Dict[str, Dict] = {}

# Create Pydantic classes
class InputText(BaseModel):
    system_prompt: str
    user_prompt: str
    model: str
    output_type: str
    image_url: str | None
    document_file: str | None

# Generate unique ids for each model request
def generate_unique_id() -> str:
    """
    Function to generate a unique ID for each request.
    Uses UUID version 4 for randomness and uniqueness.
    """
    return str(uuid.uuid4())

# Information endpoint
@app.get("/help",response_class=PlainTextResponse)
async def help():
    return """
    Hello and welcome to our amazing app! To use the endpoints of the model, you need to follow these steps:
    
    1. Send a post request with this structure

    curl -X POST "https://poc-genai.azurewebsites.net/api/process" \
     -H "Content-Type: application/json" \
     -d '{
           "system_prompt": "You are an intelligent assistant that helps generate structured data.",
           "user_prompt": "Generate a set of JSON data for a hypothetical e-commerce website. The data should include fields for product name, price, and availability.",
           "model": "llama3",
           "output_type": "JSON array"
           "image_url": "https//:example.com.png" | None
         }'

    
    2. To retrieve a past model response, send a get request with the id that was provided in the process response:
    
    curl -X GET "https://poc-genai.azurewebsites.net/api/retrieve/550e8400-e29b-41d4-a716-446655440000"
    
    Enjoy!
    """

# Endpoint to process the model request
@app.post("/api/process")
async def process(input_text:InputText):
    # Log the incoming request
    logger.info(f"Received input: {input_text}")

    # Launch classes depending on the model input
    if input_text.model == 'llama3':
        process_request = Llama3Model(input_text)
    elif input_text.model == 'llava:13b':
        process_request = LlavaModel(input_text)
    else:
        process_request = MistralModel(input_text)
    
    # Execute the request
    response = process_request.execute_model()

    return response

# Endpoint to retrieve a previous model request
@app.get("/api/retrieve/{request_id}")
async def retrieve_model(request_id: str):
    if request_id in storage:
        # Retrieve JSON string from storage
        json_content = storage[request_id]
        # Parse JSON string to Python dict
        data = json.loads(json_content)
        return {"data": data}
    else:
        raise HTTPException(status_code=404, detail="Request ID not found")


class Llama3Model:
    def __init__(self, input_text):
        self.input_text = input_text
    
    def execute_model(self):
         # Construct the content with the output type
        user_prompt = f"""
        {self.input_text.user_prompt}

        Aditional instructions:
        - Output format must be a {self.input_text.output_type}.
        - Please, follow the instructions specified in this prompt. Don't do anything that was not requested in this prompt.
        """
        client = Client(host='http://ollama:11434')
        # Prepare messages for Ollama chat
        messages = [
            {'role': 'system', 'content': self.input_text.system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

        # Call the external model
        try:
                response = client.chat(model=self.input_text.model, messages=messages)

        except Exception:
            logger.warning(f'Model is not loaded into container. Loading model {self.input_text.model}...')
            # Define the endpoint URL for the Ollama-container
            ollama_url = "http://ollama:11434/api/pull"
            # Prepare the payload with the model name
            payload = {"model": self.input_text.model}
            # Send the POST request to the Ollama-container
            requests.post(ollama_url, json=payload)
            logger.info(f'Model {self.input_text.model} loaded')
            # Retry the chat request after loading the model
            response = client.chat(model=self.input_text.model, messages=messages)

        try:
            # Obtain the response from the model
            data = response['message']['content']

            # Return data with its request_id 
            request_id = generate_unique_id()
            storage[request_id] = json.dumps(data)
            
            return {"id": request_id, "data": data}
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
class LlavaModel:
    def __init__(self, input_text):
        self.input_text = input_text
    
    def execute_model(self):
         # Construct the content with the output type
        user_prompt = f"""
        {self.input_text.user_prompt}

        Aditional instructions:
        - Output format must be a {self.input_text.output_type}.
        - Please, follow the instructions specified in this prompt. Don't do anything that was not requested in this prompt.
        """
        client = Client(host='http://ollama:11434/api/generate')

        try:
            # Download and encode the image
            image_response = requests.get(self.input_text.image_url)
            image_response.raise_for_status()
            image = Image.open(BytesIO(image_response.content))
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Prepare messages for Ollama chat
            messages = [
                {'role': 'system', 'content': self.input_text.system_prompt},
                {'role': 'user', 'content': user_prompt, 'images': [image_base64]},
            ]
        except Exception as e:
            logger.error(f"Error downloading or encoding image: {e}")
            raise HTTPException(status_code=400, detail="Error processing image URL")

        # Call the external model
        try:
                response = client.chat(model=self.input_text.model, messages=messages)

        except Exception:
            logger.warning(f'Model is not loaded into container. Loading model {self.input_text.model}...')
            # Define the endpoint URL for the Ollama-container
            ollama_url = "http://ollama:11434/api/pull"
            # Prepare the payload with the model name
            payload = {"model": self.input_text.model}
            # Send the POST request to the Ollama-container
            requests.post(ollama_url, json=payload)
            logger.info(f'Model {self.input_text.model} loaded')
            # Retry the chat request after loading the model
            response = client.chat(model=self.input_text.model, messages=messages)

        try:
            # Obtain the response from the model
            data = response['message']['content']

            # Return data with its request_id 
            request_id = generate_unique_id()
            storage[request_id] = json.dumps(data)
            
            return {"id": request_id, "data": data}
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

class MistralModel:
    vector_store = None
    retriever = None
    chain = None
    def __init__(self, input_text):
        self.input_text = input_text
    
    def execute_model(self):
        pass