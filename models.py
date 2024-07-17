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
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from utils import upload_documents, upload_index, download_index, generate_unique_id
from constants import InputText
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

        # Call the external model
        try:
                response = client.chat(model=self.input_text.model, messages=messages)

        except Exception:
            logger.warning(f'Model is not loaded into container. Loading model {self.input_text.model}...')
            # Define the endpoint URL for the Ollama-container
            ollama_url = f"{ollama_host}/api/pull"
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
        self.text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context:

            <context>
            {context}
            </context>

            Question: {input}""")
    
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
        #Loading embedding
        embeddings = OllamaEmbeddings(model=self.input_text.model, base_url=ollama_host)
        model = ChatOllama(model=self.input_text.model, base_url=ollama_host)
        try:
            if file_paths:
                counter = 0
                for file_path in file_paths:
                    # Load documents locally
                    loader = PyPDFLoader(file_path)
                    pages = loader.load_and_split()
                    # Save them permanently in Azure Container
                    upload_documents(self.input_text.username, file_path, document_files[counter].filename)
                    logger.info(f"Loaded {len(pages)} pages")
                    # Split the Text into Individual Questions
                    documents = self.text_splitter.split_documents(pages)
                    #Create vector store
                    vector_store_db = FAISS.from_documents(documents=documents, embedding=embeddings)
                    vector_store_db.save_local(folder_path="faiss_index", index_name=document_files[counter].filename)
                    # Save vector store index in Azure
                    upload_index(self.input_text.username, "faiss_index", document_files[counter].filename)
                    counter =+ 1
                return {'Documents loaded succesfully'}
            else:
                # Download index from Azure
                vector_store_db = download_index(self.input_text.username, "faiss_index", embeddings)
                logger.info('Vector store has been created')
                if self.input_text.input_choice == "Ask a question to the knowledge base":
                    # Retrieve the information
                    retriever = vector_store_db.as_retriever()
                    # Query the vector store
                    document_chain = create_stuff_documents_chain(model, self.prompt)
                    chain = create_retrieval_chain(retriever, document_chain)
                    result = chain.invoke({"input": self.input_text.user_prompt})
                    sources = []
                    for doc in result['context']:
                        sources.append({"Source":doc.metadata["source"]})
                    # Return data with its request_id 
                    request_id = generate_unique_id()
                    storage[request_id] = {'data': result["answer"], 'Sources': json.dumps(sources)}
                    return {'data': result["answer"], 'Sources': sources}
                else:
                    result = vector_store_db.similarity_search_with_score(self.input_text.user_prompt)
                    # Return data with its request_id 
                    request_id = generate_unique_id()
                    storage[request_id] = {'data': str(result)}
                    return {'data': str(result)}


        except Exception as e:
            logger.info(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    