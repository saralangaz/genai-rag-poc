from fastapi import FastAPI, HTTPException, UploadFile, Form, File
import json
import logging
from typing import Dict, Optional, List
import tempfile
import os
from dotenv import load_dotenv
from constants import gen_system_prompt, gen_text_prompt, gen_image_prompt, InputText
from utils import get_all_usernames, authenticate_user
from models import MultiModalModel, RagModel
from fastapi.responses import StreamingResponse
import asyncio

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# In-memory storage for simplicity
storage: Dict[str, Dict] = {}

async def astreamer(generator):
    try:
        for i in generator:
            yield (i)
            await asyncio.sleep(.1)
    except asyncio.CancelledError as e:
        raise ValueError

# Endpoint to process the model request
@app.post("/api/process")
async def process(model: str = Form(None),
                  use_case: str = Form(...),
                  collection_name: Optional[str] = Form(None),
                  system_prompt: Optional[str] = Form(None),
                  user_prompt: Optional[str] = Form(None),
                  image_url: Optional[str] = Form(None),
                  username: str = Form(...),
                  document_files: List[UploadFile] = File(None)):
    """
    Process a request to interact with the Ollama model based on input parameters.

    This endpoint handles HTTP POST requests sent to '/api/process' for processing
    text and/or image inputs using different models. It ensures authentication 
    using username and password stored in Cosmos DB.

    Parameters:
    -----------
    model : str
        The name of the model to be used for processing.
    use_case : str
        Specifies the use case scenario ('multimodal' or 'rag').
    system_prompt : str, optional
        The system prompt input for the model.
    user_prompt : str, optional
        The user prompt input for the model.
    image_url : str, optional
        URL of the image to be processed.
    username : str
        Username for authentication.
    input_choice : str, optional
        Type of input choice for RAG model ('question' or 'similarity search').
    document_files : List[UploadFile], optional
        List of uploaded files (PDF format) for processing.

    Returns:
    --------
    dict
        A dictionary containing the processed data or error details.

    Raises:
    -------
    HTTPException
        If authentication fails (status_code 401).
    """
    
    # Assign the received data to input text
    input_text = InputText(
        model=model,
        use_case=use_case,
        collection_name=collection_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_url=image_url,
        username=username
    )

    # Ensure the username in the request matches the authenticated user
    user_list = get_all_usernames()
    if not authenticate_user(input_text.username, user_list):
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Log the incoming request
    logger.info(f"Received input: {input_text}. Document files are {document_files}")
    temp_file_paths = []
    if document_files:
        for document_file in document_files:
            # Read the file as binary data
            file_content = await document_file.read()
            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, prefix=f"{document_file.filename}_", suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_file_paths.append(temp_file.name)

    # Launch classes depending on the model input
    if input_text.use_case == "multimodal":
        process_request = MultiModalModel(input_text, gen_system_prompt, gen_image_prompt, gen_text_prompt)
        return StreamingResponse(astreamer(process_request.execute_model(temp_file_paths, document_files)), media_type="text/event-stream")
    else:
        process_request = RagModel(input_text)
        return process_request.execute_model(temp_file_paths, document_files)

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
