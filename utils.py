from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError, ResourceNotFoundError
import os
from fastapi import HTTPException
from dotenv import load_dotenv
import logging
from azure.cosmos import CosmosClient, exceptions
from langchain_community.vectorstores import FAISS
import uuid

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure connections
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING, connection_timeout=60, read_timeout=120)
COSMOS_DB_URI = os.getenv("COSMOS_DB_URI")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
DATABASE_NAME = os.getenv("COSMOS_DB_DATABASE_NAME")
CONTAINER_NAME = os.getenv("COSMOS_DB_CONTAINER_NAME")

# Function to ensure correct base64 padding
def ensure_correct_padding(encoded_key):
    padding = len(encoded_key) % 4
    if padding != 0:
        encoded_key += '=' * (4 - padding)
    return encoded_key

def create_container_if_not_exists(container_name: str):
    try:
        container_client = blob_service_client.get_container_client(container_name)
        container_client.get_container_properties()  # Check if container exists
    except ResourceNotFoundError:
        # If container does not exist, create it
        container_client = blob_service_client.create_container(container_name)

def upload_documents(container_name: str, filepath:str, filename: str):
    try:
        # Ensure the container exists
        create_container_if_not_exists(container_name)

        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=f'documents/{filename}')

        # Upload the file to Azure Blob Storage
        blob_client.upload_blob(filepath, overwrite=True)
        
    except AzureError as e:
        raise HTTPException(status_code=500, detail=f"AzureError: {str(e)}")

def upload_index(container_name: str, faiss_index: str, filename: str):
    container_client = blob_service_client.get_container_client(container=container_name)
    with open(f'{faiss_index}/{filename}.faiss', "rb") as faiss_file:
        faiss_data = faiss_file.read()
        container_client.upload_blob(name=f'faiss_index/{filename}.faiss', data=faiss_data, overwrite=True)
    with open(f'{faiss_index}/{filename}.pkl', "rb") as faiss_pkl:
        pkl_data = faiss_pkl.read()
        container_client.upload_blob(name=f'faiss_index/{filename}.pkl', data=pkl_data, overwrite=True)

def download_index(container_name: str, faiss_index: str, embeddings):
    # Ensure download directory exists
    if not os.path.exists(faiss_index):
        os.makedirs(faiss_index)

    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs(name_starts_with=faiss_index)
    idx_names = []
    
    for blob in blobs:
        filename = os.path.split(blob.name)[-1]
        blob_client = container_client.get_blob_client(blob.name)
        with open(os.path.join(faiss_index,filename), "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        if '.faiss' in filename:
            idx_names.append(filename.removesuffix('.faiss'))
    
    vector_db = None
    # Merge all indexes
    for idx in idx_names:
        if not vector_db:
            # Load the index to FAISS db
            vector_db = FAISS.load_local("faiss_index", index_name=idx, embeddings=embeddings, allow_dangerous_deserialization=True)
        else:
            tmp_vector_db = FAISS.load_local("faiss_index", index_name=idx, embeddings=embeddings, allow_dangerous_deserialization=True)
            vector_db.merge_from(tmp_vector_db)
    
    return vector_db


def get_all_usernames():
    try:
        # Initialize the Cosmos client
        client = CosmosClient(COSMOS_DB_URI, ensure_correct_padding(COSMOS_DB_KEY))

        # Get the database
        database = client.get_database_client(DATABASE_NAME)

        # Get the container
        container = database.get_container_client(CONTAINER_NAME)

        # Query to retrieve all usernames
        query = "SELECT c.userid, c.password FROM c"
        usernames = []
        items = container.query_items(query=query, enable_cross_partition_query=True)

        for item in items:
            usernames.append((item['userid'],item['password']))

        return usernames
    except exceptions.CosmosHttpResponseError as e:
        print(f"An error occurred with Cosmos DB: {str(e)}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return []

# Generate unique ids for each model request
def generate_unique_id() -> str:
    """
    Function to generate a unique ID for each request.
    Uses UUID version 4 for randomness and uniqueness.
    """
    return str(uuid.uuid4())

# Function to authenticate user based on username
def authenticate_user(username: str, users_list: list):
    for username in users_list:
        if username:
           return True
    return False