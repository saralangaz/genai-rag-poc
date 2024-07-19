from fastapi import HTTPException
from ollama import Client
import json
import logging
import requests
from PIL import Image
from io import BytesIO
import base64
import os
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from utils import upload_documents, save_collection, load_collection, generate_unique_id
from constants import InputText
import ollama
import chromadb
load_dotenv()

# Load env variables
ollama_host = os.getenv('OLLAMA_HOST', "http://ollama:11434")
# In-memory storage for simplicity
storage: Dict[str, Dict] = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Class to upload generic inputs for the models
class GenericInputs:
    """
    Represents generic inputs for model processing.
    """
    def __init__(self, gen_system_prompt, gen_image_prompt, gen_text_prompt):
        self.gen_system_prompt = gen_system_prompt
        self.gen_image_prompt = gen_image_prompt
        self.gen_text_prompt = gen_text_prompt

# MultiModal Use Case
class MultiModalModel(GenericInputs):
    """
    Represents a multi-modal model processing instance inheriting from GenericInputs.
    """
    def __init__(self, input_text: InputText, gen_system_prompt: str, gen_image_prompt: str, gen_text_prompt: str):
        """
        Initialize the MultiModalModel with input text and generic prompts.
        """
        super().__init__(gen_system_prompt, gen_image_prompt, gen_text_prompt)
        self.input_text = input_text
    
    def execute_model(self, file_path=None, document_file=None):
        """
        Execute the multi-modal model processing with optional file inputs.

        This method prepares the input messages, including system and user prompts,
        optionally downloading and encoding an image, and interacts with an external
        model to obtain a response. It stores the processed data along with a unique
        request ID in the application's storage.

        Parameters:
        -----------
        file_path : str or None, optional
            The path to the file to be processed (default is None).
        document_file : UploadFile or None, optional
            The document file to be processed (default is None).

        Returns:
        --------
        dict
            A dictionary containing the request ID and processed data.

        Raises:
        -------
        HTTPException
            If there's an error processing the request, such as model loading failure
            or issues with the image URL (status_code 500 or 400).
        """
        if self.input_text.system_prompt is None:
            self.input_text.system_prompt = self.gen_system_prompt
        if self.input_text.user_prompt is None:
            self.input_text.user_prompt = self.gen_image_prompt
        
        logger.info(f"System prompt is {self.input_text.system_prompt} and user prompt is {self.input_text.user_prompt}")

         # Construct the content with the output type
        user_prompt = f"""
        {self.input_text.user_prompt}

        Aditional instructions:
        - Please, follow the instructions specified in this prompt. Don't do anything that was not requested in this prompt.
        """
        client = Client(host=f'{ollama_host}/api/generate')

        try:
            # Prepare message
            messages = [ {'role': 'system', 'content': self.input_text.system_prompt},
                         {'role': 'user', 'content': user_prompt}]
            # Download and encode the image
            if self.input_text.image_url:
                image_response = requests.get(self.input_text.image_url)
                image_response.raise_for_status()
                image = Image.open(BytesIO(image_response.content))
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                # Add image to the message
                messages[1]['images']= [image_base64]
        except Exception as e:
            logger.error(f"Error downloading or encoding image: {e}")
            raise HTTPException(status_code=400, detail="Error processing image URL")
        
        try:                
            # Check if model is loaded into container
            r = requests.get(f'{ollama_host}/api/tags')
            if self.input_text.model not in str(r.content):
                logger.info(f'Model is not loaded into container. Loading model...')
                # Define the endpoint URL for the Ollama-container
                ollama_url = f"{ollama_host}/api/pull"
                # Prepare the payload with the model name
                payload = {"model": self.input_text.model}
                # Send the POST request to the Ollama-container
                requests.post(ollama_url, json=payload)
                logger.info(f'Model {self.input_text.model} loaded')

            # Call the external model
            response = client.chat(model=self.input_text.model, messages=messages)
            
            # Obtain the response from the model
            data = response['message']['content']

            # Return data with its request_id 
            request_id = generate_unique_id()
            storage[request_id] = {"username": self.input_text.username, "data": json.dumps(data)}
            
            return {"id": request_id, "data": data}
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Rag Use Case
class RagModel(GenericInputs):
    """
    Represents a RAG (Retrieval-Augmented Generation) model processing instance inheriting from GenericInputs.
    """
    def __init__(self, input_text: InputText):
        """
        Initialize the RagModel with input text and necessary components.

        Parameters:
        -----------
        input_text : InputText
            The input text containing model parameters and prompts.
        """
        self.input_text = input_text

    def execute_model(self, file_paths, document_files):
        """
        Execute the RAG model processing with optional file inputs.

        This method manages document loading, indexing, querying, and response retrieval using
        the RAG model and associated components.

        Parameters:
        -----------
        file_paths : list of str or None
            List of file paths to be processed (default is None).
        document_files : list of UploadFile or None
            List of document files to be processed (default is None).

        Returns:
        --------
        dict
            A dictionary containing the processed data or confirmation of operation.

        Raises:
        -------
        HTTPException
            If there's an error processing the request (status_code 500).
        """
        client = chromadb.PersistentClient(path="chromadb/")
        # Check if model is loaded into container
        r = requests.get(f'{ollama_host}/api/tags')
        for model in [self.input_text.embed_model, self.input_text.model]:
            if model not in str(r.content):
                logger.info(f'Model is not loaded into container. Loading model...')
                # Define the endpoint URL for the Ollama-container
                ollama_url = f"{ollama_host}/api/pull"
                # Prepare the payload with the model name
                payload = {"model": model}
                # Send the POST request to the Ollama-container
                requests.post(ollama_url, json=payload)
                logger.info(f'Model {model} loaded')
        try:
            if file_paths:
                logger.info(f'Loading Files...')
                counter = 0
                for file_path in file_paths:
                    # Create chromadb collection
                    collection = client.create_collection(name=document_files[counter].filename)
                    # Load documents locally and split
                    loader = PyPDFLoader(file_path)
                    documents = loader.load_and_split()
                    # Save them permanently in Azure Container
                    upload_documents(self.input_text.username, file_path, document_files[counter].filename)
                    logger.info(f"Loaded {len(documents)} pages")
                    # store each document in a vector embedding database
                    for i, d in enumerate(documents):
                        # Convert document to a JSON-serializable format
                        document_text = d.page_content if hasattr(d, 'page_content') else str(d)
                        response = ollama.embeddings(model=self.input_text.embed_model, prompt=document_text)
                        embedding = response["embedding"]
                        collection.add(
                            ids=[str(i)],
                            embeddings=[embedding],
                            documents=[document_text]
                        )
                    # # Save vector store index in Azure
                    save_collection(self.input_text.username, "chromadb", self.input_text.collection_name, collection.get(include=['embeddings', 'documents', 'metadatas']))
                    counter =+ 1
                return {'Documents loaded succesfully'}
            else:
                try:
                    # Load collection
                    logger.info('Loading collection...')
                    collection = client.get_collection(self.input_text.collection_name,)
                except Exception as e:
                    # Download collection from Azure
                    logger.info(f'{str(e)}. Downloading collection...')
                    json_col = load_collection(self.input_text.username, "chromadb", self.input_text.collection_name,)
                    collection = client.create_collection(self.input_text.collection_name,)
                    collection.add(
                        embeddings = json_col['embeddings'],
                        documents = json_col['documents'],
                        metadatas = json_col['metadatas'],
                        ids = json_col['ids'])
                # Generate an embedding for the prompt and retrieve the most relevant doc
                response = ollama.embeddings(
                prompt=self.input_text.user_prompt,
                model=self.input_text.embed_model
                )
                results = collection.query(
                query_embeddings=[response["embedding"]],
                n_results=1
                )
                data = results['documents'][0][0]
                # Provide response
                output = ollama.generate(
                model=self.input_text.model,
                prompt=f"Using this data: {data}. Respond to this prompt: {self.input_text.user_prompt}"
                )

                # Return data with its request_id 
                request_id = generate_unique_id()
                storage[request_id] = {'data': output['response']}
                return{"id": request_id, 'data': output['response']}
        except Exception as e:
            logger.info(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            