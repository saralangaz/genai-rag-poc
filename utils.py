from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError, ResourceNotFoundError
import os
from fastapi import HTTPException
from dotenv import load_dotenv
import logging
from azure.cosmos import CosmosClient, exceptions
import uuid
import json
import re

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

def create_container_if_not_exists(container_name: str = 'images'):
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

def upload_image_to_blob(image_path, blob_container, blob_name, container_name = 'images'):
    try:
        # Check if container exists and create one if not
        create_container_if_not_exists(container_name)
        
        # Create a blob client using the container and blob name
        blob_name = f'{blob_container}/{blob_name}'
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # Upload the image
        with open(image_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        # Get the URL for the uploaded blob
        blob_url = blob_client.url
        logger.info(f"Image successfully uploaded to: {blob_url}")
        
        return blob_url
    
    except Exception as e:
        logger.error(f"Failed to upload image: {e}")
        return None

def get_image_url_from_blob(blob_container, blob_name, container_name = 'images'):
    try:
        # Get the blob client
        blob_name = f'{blob_container}/{blob_name}'
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # Return the URL for the blob
        blob_url = blob_client.url

        return blob_url
    
    except Exception as e:
        logger.error(f"Failed to retrieve image URL: {e}")
        return None

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

def filter_data(data, keywords):
    filtered_data = []

    for text in data:
        # Split text into sentences for processing
        sentences = text.split('\n')

        # Keep only the sentences that contain one or more keywords
        relevant_sentences = \
            [sentence for sentence in sentences if any(re.search(rf'\b{keyword}\b', sentence.lower()) for keyword in keywords)]
        
        # Join relevant sentences back together
        filtered_text = ". ".join(relevant_sentences)
        
        if filtered_text:
            filtered_data.append(filtered_text)
    
    return filtered_data
