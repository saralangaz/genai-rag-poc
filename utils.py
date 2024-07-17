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
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=f'documents/{filename}')

        # Upload the file to Azure Blob Storage
        blob_client.upload_blob(filepath, overwrite=True)
        
    except AzureError as e:
        raise HTTPException(status_code=500, detail=f"AzureError: {str(e)}")

def upload_index(container_name: str, faiss_index: str, filename: str):
    """
    Uploads a Faiss index and its corresponding pickle file to Azure Blob Storage.

    Parameters:
    -----------
    container_name : str
        Name of the Azure Blob Storage container where the files will be uploaded.
    faiss_index : str
        Directory path where the Faiss index files are located locally.
    filename : str
        Name to assign to the Faiss index files in Azure Blob Storage.

    Notes:
    ------
    This function assumes that the Faiss index and its corresponding pickle file
    are located in the specified `faiss_index` directory locally. It uploads these
    files to Azure Blob Storage under the 'faiss_index' directory with the specified
    `filename`.

    Example:
    --------
    >>> upload_index("my-container", "/local/path/to/faiss_index", "index1")
    """

    container_client = blob_service_client.get_container_client(container=container_name)
    with open(f'{faiss_index}/{filename}.faiss', "rb") as faiss_file:
        faiss_data = faiss_file.read()
        container_client.upload_blob(name=f'faiss_index/{filename}.faiss', data=faiss_data, overwrite=True)
    with open(f'{faiss_index}/{filename}.pkl', "rb") as faiss_pkl:
        pkl_data = faiss_pkl.read()
        container_client.upload_blob(name=f'faiss_index/{filename}.pkl', data=pkl_data, overwrite=True)

def download_index(container_name: str, faiss_index: str, embeddings):
    """
    Downloads Faiss index files from Azure Blob Storage, merges them into a single index,
    and returns the merged FAISS vector database.

    Parameters:
    -----------
    container_name : str
        Name of the Azure Blob Storage container where the index files are stored.
    faiss_index : str
        Directory path to save the downloaded Faiss index files locally.
    embeddings : object
        Embeddings object used to initialize the FAISS vector database.

    Returns:
    --------
    vector_db : FAISS
        Merged FAISS vector database loaded with the downloaded index files.

    Notes:
    ------
    This function assumes that the Faiss index files in Azure Blob Storage are under the
    'faiss_index' directory within the specified `container_name`. It downloads these files
    to the local `faiss_index` directory, merges them into a single FAISS vector database,
    and returns the merged database.

    Example:
    --------
    >>> embeddings = OllamaEmbeddings(model="my_model", base_url="https://ollama-host.com")
    >>> vector_db = download_index("my-container", "/local/path/to/faiss_index", embeddings)
    """

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