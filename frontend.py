import gradio as gr
import requests
from constants import mm_models, rag_models
import os
import logging
from dotenv import load_dotenv
from utils import get_all_usernames
import json
load_dotenv()

# Load environment variables
backend_host = os.getenv('BACKEND_HOST', "http://backend:8000")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a function to process the request
def process_mm_request(input, request: gr.Request):
    """
    Send a POST request to a FastAPI backend for processing multi-modal requests.

    Parameters:
    -----------
    model : str
        The model name or identifier.
    use_case : str
        The specific use case for the request.
    request : gr.Request
        The request object containing user information.
    system_prompt : str or None, optional
        The system prompt for the request (default is None).
    user_prompt : str or None, optional
        The user prompt for the request (default is None).
    image_url : str or None, optional
        The URL of an image related to the request (default is None).

    Returns:
    --------
    dict
        JSON response from the FastAPI backend.
    """

    try:
        url = f"{backend_host}/api/process"  # Adjust this to your FastAPI server address
        input = json.loads(input)

        if input.get("model") not in mm_models:
            return {'error': f'The model specified must be one of: {mm_models}'}
        
        payload = {
            "model": input.get("model"),
            "use_case": "multimodal",
            "collection_name": None,
            "system_prompt": input.get("system_prompt"),
            "user_prompt": input.get("user_prompt"),
            "image_url": input.get("image_url"),
            "username": request.username,
            "input_choice": None
        }
        logger.info(f'Info loaded: Payload is {payload}')
        
        # Send POST request to FastAPI backend
        response = requests.post(url, data=payload, files=None, stream=True)
        collected_data = ""
        for line in response.iter_lines():
            decoded_line = line.decode('utf-8')
            if decoded_line != None:
                collected_data += decoded_line
                yield collected_data
    except Exception as e:
            return {'error': f'{str(e)}'}
    
def process_rag_request(input: dict, request: gr.Request, files=None):
    """
    Send a POST request to a FastAPI backend for processing RAG (Retrieval-Augmented Generation) requests.

    Parameters:
    -----------
    model : str
        The model name or identifier.
    use_case : str
        The specific use case for the request.
    request : gr.Request
        The request object containing user information.
    user_prompt : str
        The user prompt for the request.
    input_choice : str
        The type of input choice (e.g., "Ask a question to the knowledge base", "Upload a Document").
    files : List[UploadFile], optional
        List of files to be uploaded for the request (default is None).

    Returns:
    --------
    dict
        JSON response from the FastAPI backend.
    """
        
    url = f"{backend_host}/api/process"  # Adjust this to your FastAPI server address
    input = json.loads(input)
    if input.get("model") not in rag_models:
            return {'error': f'The model specified must be one of: {rag_models}'}

    payload = {
        "model": input.get("model"),
        "use_case": "rag",
        "collection_name": input.get("collection_name"),
        "system_prompt": None,
        "user_prompt": input.get("user_prompt"),
        "image_url": None,
        "username": request.username,
        "input_choice": input.get("input_choice")
    }
    logger.info(f'Info loaded: Payload is {payload}')
    
    # Send POST request to FastAPI backend
    processed_files = []
    if files:
        for file in files:
            if file:
                processed_files.append(('document_files', open(file.name, 'rb')))
    # Send POST request to FastAPI backend
    response = requests.post(url, data=payload, files=None, stream=True)
    collected_data = ""
    for line in response.iter_lines():
        decoded_line = line.decode('utf-8')
        if decoded_line != None:
            collected_data += decoded_line
            yield collected_data

# Define a function to retrieve old requests
def retrieve_request(request_id):
    """
    Retrieve data associated with a specific request ID from a FastAPI backend endpoint.

    Parameters:
    -----------
    request_id : str
        The unique identifier of the request to retrieve.

    Returns:
    --------
    dict
        JSON response containing data associated with the request ID.

    Raises:
    ------
    HTTPException
        If the request fails or the request ID is not found (status_code != 200).
    """

    url = f"{backend_host}/api/retrieve/{request_id}"  # Use the Azure service name
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("Choose the Use Case for this demo.")
    # Define Gradio components for multi-modal m:odel processing
    with gr.Tab("Ask the chatbot"):
        gr.Markdown("**Multi-Modal Use Case:** This tab is used to generate random data, provide information about an image, etc. "
                    "Mandatory inputs: model type (llama2:7b, llama3:8b, llava:7b). Optional inputs: system prompt, user prompt, image URL. "
                    "Structure:")
        gr.Markdown("""```
                    {"model": "llama2:7b",
                     "system_prompt": "You are an intelligent assistant that helps generate structured data.",
                     "user_prompt":"Generate a set of JSON data for a hypothetical e-commerce website. The data should include fields for product name, price, and availability",
                     "image_url": ""}
                ```""")
        mm_input = gr.Textbox(label="Inputs")
        mm_button = gr.Button("Submit")
        mm_out = gr.Textbox(label="Model Response", lines=10)
    
    with gr.Tab("Upload documents and ask the Knowledge Base"):
        # Define Gradio components for RAG model processing
        gr.Markdown("**Rag Use Case:** This tab is used to Upload documents to a Vectorial Database and ask questions or extract information from those documents. "
                    "Mandatory inputs: model type (llama2:7b, llama3:8b, mistral:7b), collection name and input choice (Document Upload or User Query). Embeddings will be stored in a vectorial database under a collection name. "
                    "You can save all the documents you want inside the same collection name. "
                    "Structure:")
        gr.Markdown("""```
            {"model": "llama2:7b",
             "collection_name": "news_2024",
             "input_choice": "Upload one or more documents to the knowledge base" / "Ask a question to the knowledge base"
             "user_prompt": "" / "can you summarize the news of June?"}
        ```""")
        rag_input = gr.Textbox(label="Inputs")
        rag_file_input = gr.Files(label="Upload Documents", type="filepath")
        rag_out = gr.Textbox(label="Model Response", lines=10)
        rag_button = gr.Button("Submit")

    with gr.Tab("ID Request Retrieval"):
        # Define Gradio components for model retrieval
        request_id_input = gr.Textbox(label="Request ID to Retrieve", placeholder="550e8400-e29b-41d4-a716-446655440000")
        ret_button = gr.Button("Retrieve")
        ret_out = gr.Textbox(label="Model Response")
   
    mm_button.click(process_mm_request, inputs=mm_input, outputs=mm_out)
    rag_button.click(process_rag_request, inputs=[rag_input, rag_file_input], outputs=rag_out)
    ret_button.click(retrieve_request, inputs=request_id_input, outputs=ret_out)

# Launch the combined interface
demo.launch(server_name="0.0.0.0", auth=(get_all_usernames()))