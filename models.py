from fastapi import HTTPException
import json
import logging
import requests
from PIL import Image
from io import BytesIO
import base64
import os
from constants import ExecuteModelInput, CollectionInput, ImageInput, IMAGE_DIR, gen_system_prompt
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
import clip
import torch
from sklearn.decomposition import PCA

load_dotenv()

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

# Load env variables
ollama_host = os.getenv('OLLAMA_HOST', "http://ollama:11434")
backend_host = os.getenv('BACKEND_HOST', "http://backend:8000") 
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
    
    def ensure_collection(self, collection_name: str):
        """
        Ensure the Weaviate collection exists and has both text and image properties.
        """
        try:
            # Check if the collection already exists
            collection = client.collections.get(collection_name)
            return collection

        except Exception as e:
            if '404' in str(e):
                # If the collection does not exist, create it with both text and image properties
                logger.info(f"Creating collection '{collection_name}'...")
                collection = client.collections.create(
                    name=collection_name,
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="image_description", data_type=DataType.TEXT),
                        Property(name="image_url", data_type=DataType.TEXT),
                        Property(name="image_name", data_type=DataType.TEXT)
                    ]
                )
                logger.info(f"Collection '{collection_name}' created with text and image properties.")
                return collection
            else:
                logger.error(f"Error processing collection: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
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
    
    def retrieve_response(self, input_text:ExecuteModelInput, data, type, image_name):
            if type == 'vision':
                logger.info('Calling vision model...')
                image_path = os.path.join(IMAGE_DIR, image_name)
                with open(image_path, 'rb') as img_file:
                    image = Image.open(img_file)
                    # Convert the image to bytes using BytesIO
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    buffered.seek(0)
                    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                # Prepare message
                messages = [ {'role': 'system', 'content': gen_system_prompt},
                            {'role': 'user', 'content': input_text.user_prompt, 'image': image_base64}]
                # Add image to the message
                output = ollama.generate(
                model="llava:7b",
                prompt=input_text.user_prompt,
                images=[image_base64],
                stream=True)
            else:
                output = ollama.generate(
                model=input_text.model,
                prompt=f"Using this data: {data}, respond to this prompt: {input_text.user_prompt}",
                stream=True)
            
            return output
    
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
            collection = self.ensure_collection(input_text.collection_name)
            for file_path in file_paths:
                # Load documents and split
                loader = PdfReader(file_path)
                documents = [p.extract_text().strip() for p in loader.pages]
                # Filter the empty strings
                documents = [text for text in documents if text]
                with collection.batch.dynamic() as batch:
                    for i, d in enumerate(documents):
                        # Generate embeddings
                        response = ollama.embeddings(model = "nomic-embed-text",
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

    def get_image_embedding(self, image_name):
        """
        Function to extract image embedding using CLIP model.
        """
        logger.info('Retrieving image...')
        image_path = os.path.join(IMAGE_DIR, image_name)
        image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = clip_model.encode_image(image).cpu().numpy().flatten()
        return image_embedding

    def load_images(self, image_paths: list, image_names: list, input_text:ImageInput):
        try:
            logger.info(f'Loading collection...')

            # Get or create the Weaviate collection
            collection = self.ensure_collection(input_text.collection_name)

            logger.info(f'Loading Images...')

            # Process images and add to Weaviate
            with collection.batch.dynamic() as batch:
                counter = 0
                for image_path in image_paths:
                    # Assuming CLIP or another model for generating image embeddings
                    image_embedding = self.get_image_embedding(image_names[counter])
                    logger.info(f"Image embedding shape: {image_embedding.shape}")  # This should print (1024,)
                    
                    with collection.batch.dynamic() as batch:
                        batch.add_object(
                            properties={"image_description": input_text.image_description,
                                        "image_url": image_path,
                                        "image_name": image_names[counter]},
                            vector=image_embedding
                        )
                    counter =+1
            return 'Images uploaded successfully'
        
        except Exception as e:
            logger.error(f"Error processing image {image_paths}: {e}")

    
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
            model = "nomic-embed-text",
            prompt = input_text.user_prompt,
            )
            embed_end = timeit.timeit()

            # Check if the prompt refers to an image or images
            image_synonims = ['image', 'picture', 'logo', 'photo']
            is_image_query = [i for i in image_synonims if i in input_text.user_prompt.lower()]

            query_start = timeit.timeit()
            name=None
            if is_image_query:
                # Query for images using the text embedding
                logger.info('Performing image query using text embedding...')
                results = collection.query.near_vector(near_vector=response['embedding'], limit=input_text.k_value)
                if 'image_description' in results.objects[0].properties:
                    data = results.objects[0].properties['image_description']
                    name = results.objects[0].properties['image_name']
                    type = 'vision'
                else:
                    data = "No image data found."
                    type = 'vision'
                    name = 'No image data found. '

            else:
                # Query for text using the text embedding
                logger.info('Performing text query...')
                results = collection.query.near_vector(near_vector=response['embedding'], limit=input_text.k_value)
                if 'text' in results.objects[0].properties:
                    data = results.objects[0].properties['text']
                    type = 'text'
                else:
                    data = "No text data found."
                    type = 'text'

            query_end = timeit.timeit()
            # Provide response
            resp_start = timeit.timeit()
            output = self.retrieve_response(input_text, data, type, name)
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
