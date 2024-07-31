from fastapi import HTTPException
import json
import logging
import requests
from PIL import Image
from io import BytesIO
import base64
import os
from typing import Dict
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from dotenv import load_dotenv
from utils import upload_documents, save_collection, load_collection, generate_unique_id, delete_documents
from constants import InputText
import ollama
import timeit
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
import weaviate
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
            # Call the external model
            response = ollama.chat(model=self.input_text.model, messages=messages, stream=True)
            
            # Obtain the response from the model
            text_response = ""
            collected_data = ""
            for chunk in response:
                content = chunk['message']['content']
                if content != None:
                    text_response += content
                    collected_data = content
                else:
                    request_id = generate_unique_id()
                    collected_data = {"id": request_id, "data": text_response}
                
                yield collected_data
        
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
        client = weaviate.connect_to_local('weaviate')
        try:
            if self.input_text.use_case == 'upload':
                logger.info(f'Loading Files...')
                # Create weaviate collection
                collection = client.collections.create(
                name =self.input_text.collection_name, # Name of the data collection
                properties=[
                    Property(name="text", data_type=DataType.TEXT), # Name and data type of the property
                    ],
                )
                for file_path in file_paths:
                    # Load documents and split
                    loader = PdfReader(file_path)
                    documents = [p.extract_text().strip() for p in loader.pages]
                    # Filter the empty strings
                    documents = [text for text in documents if text]
                    with collection.batch.dynamic() as batch:
                        for i, d in enumerate(documents):
                            # Generate embeddings
                            response = ollama.embeddings(model = "all-minilm",
                                                        prompt = d)
                            # Add data object with text and embedding
                            batch.add_object(
                                properties = {"text" : d},
                                vector = response["embedding"],
                            )
                    client.close()
                    return 'Documents uploaded'
            elif self.input_text.use_case == 'delete':
                # Delete Collection
                logger.info('Deleting collection...')
                client.collections.delete(self.input_text.collection_name)
                client.close()
                return 'Collection deleted'
            else:
                logger.info('Querying collection...')
                collection = client.collections.get(self.input_text.collection_name)
                # Generate an embedding for the prompt and retrieve the most relevant doc
                response = ollama.embeddings(
                model = "all-minilm",
                prompt = self.input_text.user_prompt,
                )
                results = collection.query.near_vector(near_vector = response["embedding"],
                                                    limit = 1)
                data = results.objects[0].properties['text']
                # Provide response
                output = ollama.generate(
                model=self.input_text.model,
                prompt=f"Using this data: {data}. Respond to this prompt: {self.input_text.user_prompt}"
                )
                client.close()
                return output['response']
        except Exception as e:
            logger.info(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            