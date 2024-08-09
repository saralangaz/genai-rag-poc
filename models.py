from fastapi import HTTPException
import json
import logging
import requests
from PIL import Image
from io import BytesIO
import base64
import os
from constants import ExecuteModelInput, CollectionInput
from typing import Dict
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from utils import generate_unique_id
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

# Instantiate at init
client = weaviate.connect_to_local('weaviate')

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
    def __init__(self, input_text: ExecuteModelInput, gen_system_prompt: str, gen_image_prompt: str, gen_text_prompt: str):
        """
        Initialize the MultiModalModel with input text and generic prompts.
        """
        super().__init__(gen_system_prompt, gen_image_prompt, gen_text_prompt)
        self.input_text = input_text
    
    def execute_model(self):
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
            for chunk in response:
                content = chunk['message']['content']
                yield content
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Rag Use Case
class RagModel(GenericInputs):
    """
    Represents a RAG (Retrieval-Augmented Generation) model processing instance inheriting from GenericInputs.
    """
    def __init__(self):
        """
        Initialize the RagModel with input text and necessary components.
        """
    
    def list_collections(self):
        """
        This method list all collections from to Weaviate Vectorial DB.

        Returns:
        --------
        str
            A string with a list of the collections.

        Raises:
        -------
        HTTPException
            If there's an error processing the request (status_code 500).
        """
        
        try:
            # List all collections
            logger.info('Listing collections...')
            response = client.collections.list_all(simple=True)
            names = [config.name for config in response.values()]
            return f'Collection list: {names}'
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    def delete_collection(self, input_text: CollectionInput):
        """
        This method deletes a collection from to Weaviate Vectorial DB.

        Parameters:
        --------
        input_text : CollectionInput
            The input text containing input parameters.

        Returns:
        --------
        str
            A string confirming the operation.

        Raises:
        -------
        HTTPException
            If there's an error processing the request (status_code 500).
        """
        
        try:
            # Delete Collection
            logger.info('Deleting collection...')
            client.collections.delete(input_text.collection_name)
            return 'Collection deleted'
       
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def load_documents(self, file_paths, input_text:CollectionInput):
        """
        This method generates embeddings and manages uploading of embedded documents to Weaviate Vectorial DB.

        Parameters:
        -------
        file_paths
            a list of paths to load the documents.
        input_text : CollectionInput
            The input text containing input parameters.

        Returns:
        --------
        str
            A string confirming the operation.

        Raises:
        -------
        HTTPException
            If there's an error processing the request (status_code 500).
        """
        
        try:
            logger.info(f'Loading Files...')
            # Get or create weaviate collection
            collection = client.collections.get(input_text.collection_name)
        except Exception as e:
            if '404' in str(e):
                logger.info(f'Creating collection...')
                collection = client.collections.create(
                name =input_text.collection_name, # Name of the data collection
                properties=[
                    Property(name="text", data_type=DataType.TEXT), # Name and data type of the property
                    ],
                )
            else:
                logger.error(f"Error processing request: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        try:
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
                return 'Documents uploaded'
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def execute_model(self, input_text: ExecuteModelInput):
        """
        Execute the RAG model processing with optional file inputs.

        This method manages indexing, querying, and response retrieval using
        the RAG model and associated components.

        Parameters:
        --------
        input_text : ExecuteModelInput
            The input text containing input parameters.

        Returns:
        --------
        str
            A string containing the processed data.

        Raises:
        -------
        HTTPException
            If there's an error processing the request (status_code 500).
        """
        try:
            logger.info('Querying collection...')
            collection = client.collections.get(input_text.collection_name)
            # Generate an embedding for the prompt and retrieve the most relevant doc
            embed_start = timeit.timeit()
            response = ollama.embeddings(
            model = "all-minilm",
            prompt = input_text.user_prompt,
            )
            embed_end = timeit.timeit()
            query_start = timeit.timeit()
            results = collection.query.near_vector(near_vector = response["embedding"],
                                                limit = input_text.k_value)
            data = results.objects[0].properties['text']
            query_end = timeit.timeit()
            # Provide response
            resp_start = timeit.timeit()
            output = ollama.generate(
            model=input_text.model,
            prompt=f"Using this data: {data}. Respond to this prompt: {input_text.user_prompt}",
            stream=True
            )
            resp_end = timeit.timeit()
            logger.info('Response timings are:')
            logger.info(f'embed_timing: {(embed_end - embed_start)} seconds')
            logger.info(f'query_timing: {(query_end - query_start)} seconds')
            logger.info(f'response_model_timing: {(resp_end - resp_start)} seconds')
            # Obtain the response from the model
            for chunk in output:
                content = chunk['response']
                yield content
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
