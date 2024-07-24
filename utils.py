from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError, ResourceNotFoundError
import os
from fastapi import HTTPException
from dotenv import load_dotenv
import logging
from azure.cosmos import CosmosClient, exceptions
import uuid
import json

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
    """
    Ensure that the Base64 encoded key has correct padding with '=' characters.

    Parameters:
    -----------
    encoded_key : str
        The Base64 encoded key that may require padding.

    Returns:
    --------
    str
        The input `encoded_key` with correct padding added, if necessary.

    Notes:
    ------
    Base64 encoded strings are padded with '=' characters to ensure they have a length
    that is a multiple of 4. This function checks the length of `encoded_key` and adds
    the appropriate number of '=' characters to ensure correct padding.

    Example:
    --------
    >>> encoded_key = 'YWJjZA'  # Example Base64 encoded string without padding
    >>> ensure_correct_padding(encoded_key)
    'YWJjZA=='
    """

    padding = len(encoded_key) % 4
    if padding != 0:
        encoded_key += '=' * (4 - padding)
    return encoded_key

def create_container_if_not_exists(container_name: str):
    """
    Create a blob storage container if it does not already exist.

    Parameters:
    -----------
    container_name : str
        Name of the container to be created or checked.

    Raises:
    -------
    ResourceNotFoundError
        If the container does not exist and cannot be created.

    Notes:
    ------
    This function checks if a blob storage container with the given `container_name` exists.
    If the container does not exist, it creates it using the `blob_service_client`.

    Example:
    --------
    >>> create_container_if_not_exists("my-container")
    """

    try:
        container_client = blob_service_client.get_container_client(container_name)
        container_client.get_container_properties()  # Check if container exists
    except ResourceNotFoundError:
        # If container does not exist, create it
        container_client = blob_service_client.create_container(container_name)

def upload_documents(container_name: str, filepath:str, filename: str):
    """
    Uploads a document file to Azure Blob Storage.

    Parameters:
    -----------
    container_name : str
        Name of the Azure Blob Storage container where the file will be uploaded.
    filepath : str
        Local filepath of the document file to upload.
    filename : str
        Name to assign to the file in Azure Blob Storage.

    Raises:
    -------
    HTTPException
        If an AzureError occurs during the upload process.

    Notes:
    ------
    This function ensures that the specified Azure Blob Storage container exists
    by calling `create_container_if_not_exists`. It then uploads the file located
    at `filepath` to Azure Blob Storage under the 'documents' directory with the
    specified `filename`.

    Example:
    --------
    >>> upload_documents("my-container", "/local/path/to/file.pdf", "file.pdf")
    """

    try:
        # Ensure the container exists
        create_container_if_not_exists(container_name)

        # Create a blob client using the local file name as the name for the blob
        container_client = blob_service_client.get_container_client(container=container_name)
        with open(filepath, mode="rb") as data:
            container_client.upload_blob(name=f'documents/{filename}', data=data, overwrite=True)
        
    except AzureError as e:
        raise HTTPException(status_code=500, detail=f"AzureError: {str(e)}")

def save_collection(container_name: str, db_path: str, collection_name: str, chroma_json_object: dict):
    """
    Save a collection (in JSON format) to Azure Blob Storage.

    This function uploads a collection, which is represented as a JSON object, to a specified Azure Blob Storage container. 
    The collection is saved with a specific path and filename.

    Parameters
    ----------
    container_name : str
        The name of the Azure Blob Storage container where the collection will be saved.
    
    db_path : str
        The path within the container where the collection will be stored.
    
    collection_name : str
        The name of the collection file to be created in the specified path. The file will be saved with a `.json` extension.
    
    chroma_json_object : dict
        The collection data to be saved. It should be a dictionary that will be converted to JSON format before uploading.

    """

    container_client = blob_service_client.get_container_client(container=container_name)
    container_client.upload_blob(name=f'{db_path}/{collection_name}', data=json.dumps(chroma_json_object), overwrite=True)

def load_collection(container_name: str, db_path: str, collection_name: str):
    """
    Load a collection from Azure Blob Storage.

    This function downloads a collection, represented as a JSON object, from a specified Azure Blob Storage container.
    The collection is loaded from a specific path and filename.

    Parameters
    ----------
    container_name : str
        The name of the Azure Blob Storage container from which the collection will be downloaded.
    
    db_path : str
        The path within the container where the collection is stored. This path will also be created locally if it does not exist.
    
    collection_name : str
        The name of the collection file to be downloaded from the specified path. The file is expected to be in JSON format.
    
    Returns
    -------
    dict
        The collection data loaded from the Azure Blob Storage, returned as a dictionary.
    """
    
    # Ensure download directory exists
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=f'{db_path}/{collection_name}')
    downloader = blob_client.download_blob(max_concurrency=1, encoding='UTF-8')
    blob_text = downloader.readall()
    return json.loads(blob_text)

def get_all_usernames():
    """
    Retrieves all usernames and corresponding passwords from Cosmos DB.

    Returns:
    --------
    list
        A list of tuples containing (username, password).

    Raises:
    ------
    exceptions.CosmosHttpResponseError
        If there is an error response from Cosmos DB.
    Exception
        For any other unexpected errors.
    """

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
        logger.error(f"An error occurred with Cosmos DB: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
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
    """
    Authenticate a user based on username against a list of user credentials.

    Parameters:
    -----------
    username : str
        The username to authenticate.
    users_list : list
        A list of tuples where each tuple contains (userid, password) pairs.

    Returns:
    --------
    bool
        True if the username exists in the users_list, False otherwise.
    """

    for username in users_list:
        if username:
           return True
    return False