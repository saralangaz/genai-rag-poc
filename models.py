from fastapi import HTTPException
import json
import logging
import requests
from PIL import Image
from io import BytesIO
import base64
import os
from constants import ExecuteModelInput, CollectionInput, ImageInput, IMAGE_DIR, stopwords_list
from typing import Dict
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from utils import upload_image_to_blob, get_image_url_from_blob, filter_data
import ollama
import timeit
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import Filter
import weaviate
import clip
import torch

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
            logger.info(f'Retrieving collection...')
            collection = client.collections.get(collection_name)
            logger.info(f'Collection is {collection}')
            return collection

        except Exception as e:
            if '404' in str(e):
                # If the collection does not exist, create it with both text and image properties
                logger.info(f"Creating collection '{collection_name}'...")
                collection = client.collections.create(
                    name=collection_name,
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="image_description", data_type=DataType.TEXT),
                        Property(name="image_url", data_type=DataType.TEXT),
                        Property(name="image_name", data_type=DataType.TEXT)
                    ],
                    vectorizer_config=[Configure.NamedVectors.text2vec_ollama(
                        name='ollama_vector',
                        api_endpoint=ollama_host,
                        model='nomic-embed-text'
                    )])
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
    
    def retrieve_response(self, input_text:ExecuteModelInput, data, type, name):
            # Join all the texts into a single string
            data = "\n\n---\n\n".join(data)
            logger.info(f'Data is {data}')
            if type == 'vision':
                images_list = []
                for image in name:
                    logger.info('Encoding image to base64...')
                    image_path = get_image_url_from_blob(input_text.collection_name, image)
                    response = requests.get(image_path)
                    image = Image.open(BytesIO(response.content))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_bytes = buffered.getvalue()
                    images_list.append(base64.b64encode(img_bytes).decode("utf-8"))
                name = "\n\n---\n\n".join(name) 
                # Create prompt
                prompt = f'''Using this data {data} and these image names {name}, respond to this question {input_text.user_prompt}.
                            There is also one or more images attached, you should have it into account to respond to the question provided.'''
                logger.info('Calling vision model...')
                output = ollama.generate(
                model="llava:7b",
                prompt=prompt,
                images=images_list)
            else:
                name = "\n\n---\n\n".join(name) 
                logger.info('Calling large language model...')
                output = ollama.generate(
                model=input_text.model,
                prompt=f'''Using this data: {data}, respond to this prompt: {input_text.user_prompt}. 
                Add the sources of the data used at the end of the response. The sources are contained in {name}.
                The structure of the output should be like the following:
                Retrieved information:
                [response]
                
                Sources:
                - [source 1]
                - [source 2]
                - ...
                Do not duplicate sources and add the page where the retrieved information has been found''')
            
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
                            properties = {"text" : d, "source": file_path},
                        )
                logger.info(f'Failed objects are {collection.batch.failed_objects}')

                return 'Documents uploaded'
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_image_embedding(self, image_name):
        """
        Function to extract image embedding using CLIP model.
        """
        logger.info('Generating embedding for image...')
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
                    # Upload the image to Blob Storage and get the blob URL
                    image_name = image_names[counter]
                    blob_url = upload_image_to_blob(os.path.join(IMAGE_DIR, image_names[counter]), input_text.collection_name, image_name)

                    # Assuming CLIP or another model for generating image embeddings
                    image_embedding = self.get_image_embedding(image_names[counter])
                    logger.info(f"Image embedding shape: {image_embedding.shape}") 
                    
                    with collection.batch.dynamic() as batch:
                        batch.add_object(
                            properties={"image_description": input_text.image_description,
                                        "image_url": blob_url,
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
            embed_start = timeit.default_timer()
            embed_end = timeit.default_timer()

            # Check if the prompt refers to an image or images
            image_synonims = ['image', 'picture', 'logo', 'photo']
            is_image_query = [i for i in image_synonims if i in input_text.user_prompt.lower()]

            # Create a filter condition to obtain only relevant image metadata 
            keywords = [word for word in input_text.user_prompt.lower().split() if word not in stopwords_list]
            logger.info(f'Query keywords are {keywords}')
            
            query_start = timeit.default_timer()
            data = []
            name = []

            if is_image_query:
                # Query for images using the text embedding
                logger.info('Performing image query using text embedding...')
                results = collection.query.near_text(query=input_text.user_prompt, limit=input_text.k_value,
                                                     filters=Filter.by_property('image_description').contains_any(keywords))
                type = 'vision'
                for result in results.objects:
                    # Append each retrieved text to the all_data list
                    if result.properties.get('image_description') is not None:
                        data.append(result.properties['image_description'])
                    if result.properties.get('image_name') is not None:
                        name.append(result.properties['image_name'])
            else:
                # Query for text using the text embedding
                logger.info('Performing text query...')
                results = collection.query.near_text(query=input_text.user_prompt, limit=input_text.k_value,
                                                     filters=Filter.by_property('text').contains_any(keywords))
                type = 'text'
                for result in results.objects:
                    # Append each retrieved text to the all_data list
                    if result.properties.get('text') is not None:
                        data.append(result.properties['text'])
                    if result.properties.get('source') is not None:
                        name.append(result.properties['source'])
                # data = filter_data(data, keywords)

            query_end = timeit.default_timer()
            # Provide response
            resp_start = timeit.default_timer()
            output = self.retrieve_response(input_text, data, type, name)
            resp_end = timeit.default_timer()

            # Log all timings
            logger.info('Response timings are:')
            logger.info(f'Embed timing: {(embed_end - embed_start):.4f} seconds')
            logger.info(f'Query timing: {(query_end - query_start):.4f} seconds')
            logger.info(f'Model response timing: {(resp_end - resp_start):.4f} seconds')
            
            return output['response']
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
