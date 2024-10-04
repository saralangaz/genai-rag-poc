from fastapi import FastAPI, HTTPException, UploadFile, Form, File
from fastapi.responses import FileResponse
import json
import logging
from typing import Dict, List
import tempfile
from dotenv import load_dotenv
from constants import gen_system_prompt, gen_text_prompt, gen_image_prompt, ExecuteModelInput, CollectionInput, ValidModels, ImageInput, \
    IMAGE_DIR
# from utils import get_all_usernames, authenticate_user
from models import MultiModalModel, RagModel
from fastapi.responses import StreamingResponse
import asyncio
import os
import shutil

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# In-memory storage for simplicity
storage: Dict[str, Dict] = {}

# Checks before initiation
# Ensure the username in the request matches the authenticated user
# user_list = get_all_usernames()
# if not authenticate_user(input_text.username, user_list):
#     raise HTTPException(status_code=401, detail="Not authenticated")


async def astreamer(generator):
    try:
        for i in generator:
            yield (i)
            await asyncio.sleep(.1)
    except asyncio.CancelledError as e:
        raise ValueError

# Endpoint to execute the model request
@app.post("/api/execute_model")
async def execute_model(input_text:ExecuteModelInput):
    """
    Process a request to interact with the Ollama model based on input parameters.

    This endpoint handles HTTP POST requests sent to '/api/execute_model' for processing
    text and/or image inputs using different models. It ensures authentication 
    using username and password stored in Cosmos DB.

    Parameters:
    -----------
    input_text : BaseModel Class
        ExecuteModelInput class

    Returns:
    --------
    str
        A string containing the processed data or error details.

    Raises:
    -------
    HTTPException
        If authentication fails (status_code 401).
    """
    
    try:
        # Check if model is valid
        if input_text.model not in ValidModels.models:
            return {'error': f'The model specified must be one of: {ValidModels.models}'}
        # Log the incoming request
        logger.info(f"Received input: {input_text}.")

        # Launch classes depending on the model input
        if input_text.use_case == "multimodal":
            process_request = MultiModalModel(input_text, gen_system_prompt, gen_image_prompt, gen_text_prompt)
            return StreamingResponse(astreamer(process_request.execute_model()), media_type="text/event-stream")
        else:
            process_request = RagModel()
            return StreamingResponse(astreamer(process_request.execute_model(input_text)), media_type="text/event-stream")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Endpoint to list all collections from DB
@app.get("/api/list_collections")
async def list_collections():
    """
    List all collections from the database.

    This endpoint handles HTTP GET requests sent to '/api/list_collections' 
    to retrieve a list of all collections available in the database. 

    Returns:
    --------
    list
        A list of dictionaries where each dictionary represents a collection 
        with its details, such as collection_id and name.

    Raises:
    -------
    HTTPException
        If an error occurs while fetching the collections (status_code 500).
    """
    try:
        # Launch class
        process_request = RagModel()
        return process_request.list_collections()
    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to delete a collection from DB
@app.post("/api/delete_collection")
async def delete_collection(input_text: CollectionInput):
    """
    Delete a specified collection from the database.

    This endpoint handles HTTP POST requests sent to '/api/delete_collection' 
    to delete a collection identified by the provided input parameters.

    Parameters:
    -----------
    input_text : CollectionInput
        A data model instance containing details required to identify 
        the collection to be deleted, such as collection_id.

    Returns:
    --------
    dict
        A dictionary containing the status and message about the deletion 
        operation.

    Raises:
    -------
    HTTPException
        If the collection could not be found or deleted (status_code 404 or 500).
    """
    try:
        # Launch class
        process_request = RagModel()
        return process_request.delete_collection(input_text)
    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to load documents to Weaviate DB
@app.post("/api/load_documents")
async def load_documents(
    collection_name: str = Form(...),
    document_files: List[UploadFile] = File(...)):
    """
    Load documents into a specified collection in the database.

    This endpoint handles HTTP POST requests sent to '/api/load_documents'
    for uploading and processing documents. The documents are associated 
    with a specified model and collection name.

    Parameters:
    -----------
    model : str
        The name of the model to use for processing the documents.
    collection_name : str
        The name of the collection where the documents will be stored.
    document_files : List[UploadFile]
        A list of files to be uploaded, where each file is an instance of 
        UploadFile. The files are expected to be in binary format, typically PDFs.

    Returns:
    --------
    dict
        A dictionary containing the status of the upload operation, 
        including success or error details.

    Raises:
    -------
    HTTPException
        If an error occurs during file upload or processing (status_code 500).
    """

    try:
        # Assign the received data to input_text
        input_text = CollectionInput(
            collection_name=collection_name
        )
        # Log the incoming request
        logger.info(f"Received input: {input_text}. Document files are {document_files}")
        temp_file_paths = []
        for document_file in document_files:
            # Read the file as binary data
            file_content = await document_file.read()
            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, prefix=f"{document_file.filename}_", suffix=".pdf") as temp_file:
                    temp_file.write(file_content)
                    temp_file_paths.append(temp_file.name)
        process_request = RagModel()
        
        return process_request.load_documents(temp_file_paths, input_text)
    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load_images")
async def load_images(
    collection_name: str = Form(...),
    image_description: str = Form(...),
    image_files: List[UploadFile] = File(...)):

    try:
        # Assign the received data to input_text
        input_text = ImageInput(
            collection_name=collection_name,
            image_description=image_description
        )
        # Log the incoming request
        logger.info(f"Received input: {input_text}. Image files are {image_files}")
        
        file_paths = []
        image_names = []
        for image_file in image_files:
           # Generate a path for the image
            file_path = os.path.join(IMAGE_DIR, image_file.filename)

            # Save the uploaded image
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)

            # Generate the image URL
            image_url = f"http://localhost:8000/images/{image_file.filename}"
            file_paths.append(image_url)
            image_names.append(image_file.filename)
        
        # Process the request with RagModel, similar to document loading
        process_request = RagModel()
        return process_request.load_images(file_paths, image_names, input_text)
    
    except Exception as e:
        # Handle exceptions
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    file_path = os.path.join(IMAGE_DIR, image_name)

    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "Image not found"}

# Endpoint to retrieve a previous model request
@app.get("/api/retrieve/{request_id}")
async def retrieve_model(request_id: str):
    """
    Retrieve processed data associated with a specific request ID.

    This endpoint handles HTTP GET requests sent to '/api/retrieve/{request_id}'
    for retrieving processed data stored in the application's storage.

    Parameters:
    -----------
    request_id : str
        The unique identifier of the request whose data is to be retrieved.

    Returns:
    --------
    dict
        A dictionary containing the retrieved processed data.

    Raises:
    -------
    HTTPException
        If the specified request ID is not found in the storage (status_code 404).
    """
    
    if request_id in storage:
        # Retrieve JSON string from storage
        json_content = storage[request_id]
        # Parse JSON string to Python dict
        data = json.loads(json_content)
        return {"data": data}
    else:
        raise HTTPException(status_code=404, detail="Request ID not found")
