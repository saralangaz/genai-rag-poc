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
load_dotenv()

# Load env variables
ollama_host = os.getenv('OLLAMA_HOST', "http://ollama:11434")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# In-memory storage for simplicity
storage: Dict[str, Dict] = {}

# Endpoint to process the model request
@app.post("/api/process")
async def process(model: str = Form(...),
                  use_case: str = Form(...),
                  system_prompt: Optional[str] = Form(None),
                  user_prompt: Optional[str] = Form(None),
                  image_url: Optional[str] = Form(None),
                  username: str = Form(...),
                  input_choice: Optional[str] = Form(None),
                  document_files: List[UploadFile] = File(None)):
    
    # Assign the received data to input text
    input_text = InputText(
        model=model,
        use_case=use_case,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_url=image_url,
        username=username,
        input_choice=input_choice
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
    else:
        process_request = RagModel(input_text)
    # Execute the request
    response = process_request.execute_model(temp_file_paths, document_files)

    return response

# Endpoint to retrieve a previous model request
@app.get("/api/retrieve/{request_id}")
async def retrieve_model(request_id: str):
    if request_id in storage:
        # Retrieve JSON string from storage
        json_content = storage[request_id]
        # Parse JSON string to Python dict
        data = json.loads(json_content)
        return {"data": data}
    else:
        raise HTTPException(status_code=404, detail="Request ID not found")
