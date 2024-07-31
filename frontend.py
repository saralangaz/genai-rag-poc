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
    input : dict
        The values to send to the backend.
    request: gr.Request
        The username.

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
            "username": request.username
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
    
def process_rag_request(input: dict, request: gr.Request, files: list =None):
    """
    Send a POST request to a FastAPI backend for processing RAG (Retrieval-Augmented Generation) requests.

    Parameters:
    -----------
    input : dict
        The values to send to the backend.
    request: gr.Request
        The username.
    files : List[UploadFile], optional
        List of files to be uploaded for the request (default is None).

    Returns:
    --------
    dict
        JSON response from the FastAPI backend.
    """
        
    try:
        url = f"{backend_host}/api/process"  # Adjust this to your FastAPI server address
        input = json.loads(input)
        usecase = 'ask'
        if input.get("model") not in rag_models:
                return {'error': f'The model specified must be one of: {rag_models}'}

        # Prepare files
        processed_files = []
        if files:
            usecase = 'upload'
            for file in files:
                if file:
                    processed_files.append(('document_files', open(file.name, 'rb')))
        payload = {
            "model": input.get("model"),
            "use_case": usecase,
            "collection_name": input.get("collection_name"),
            "system_prompt": None,
            "user_prompt": input.get("user_prompt"),
            "image_url": None,
            "username": request.username
        }

        logger.info(f'Info loaded: Payload is {payload}')
        # Send POST request to FastAPI backend
        response = requests.post(url, data=payload, files=processed_files)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    
    except Exception as e:
        return {'error': f'{str(e)}'}

def delete_collection_request(input: str, request: gr.Request):
    """
    Send a POST request to a FastAPI backend for deleting a Collection from a Vectorial DB.

    Parameters:
    -----------
    input : str
        The collection Name to be deleted.
    request: gr.Request
        The username.

    Returns:
    --------
    dict
        JSON response from the FastAPI backend.
    """
        
    try:
        url = f"{backend_host}/api/process"  # Adjust this to your FastAPI server address

        payload = {
            "model": None,
            "use_case": "delete",
            "collection_name": input,
            "system_prompt": None,
            "user_prompt": None,
            "image_url": None,
            "username": request.username
        }
        logger.info(f'Info loaded: Payload is {payload}')
        
        # Send POST request to FastAPI backend
        response = requests.post(url, data=payload, files=None)
        return response.text
    
    except Exception as e:
        return {'error': f'{str(e)}'}
    

css = """
.upload-section {
    width: 100%;
    max-width: 600px; /* Adjust the max-width as needed */
}
"""

# Create Gradio interface
with gr.Blocks(css=css) as demo:
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
    
    with gr.Tab("Ask the Knowledge Base"):
        # Define Gradio components for RAG model processing
        gr.Markdown("**Rag Use Case:** This tab is used to Upload documents to a Vectorial Database and ask questions or extract information from those documents. "
                    "Mandatory inputs: model type (llama2:7b, llama3:8b, mistral:7b), collection name and user prompt. Optional inputs: Documents to upload. Embeddings will be stored in a vectorial database under a collection name. "
                    "You can save all the documents you want inside the same collection name. "
                    "Structure:")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Send inputs to the model")
                gr.Markdown("""```
                {"model": "llama2:7b",
                "collection_name": "news_2024",
                "user_prompt": "can you summarize the news of June?"}
                ```""")
                rag_input = gr.Textbox(label="Inputs", lines=4)
        rag_button = gr.Button("Submit")
        rag_out = gr.Textbox(label="Model Response", lines=6)

    with gr.Tab("Handle Vectorial Database"):
        with gr.Row():
            with gr.Column(scale=1,):
                gr.Markdown("### Upload Documents to Vectorial DB")
                gr.Markdown("""```
                {"model": "llama2:7b",
                "collection_name": "news_2024"}
                ```""")
                rag_textfile_input = gr.Textbox(label="Inputs", lines=4)
                rag_file_input = gr.Files(label="Upload Documents", type="filepath")
                upl_button = gr.Button("Upload")
            
            with gr.Column(scale=1):
                gr.Markdown("### Delete Collection from Vectorial DB")
                delete_file_name = gr.Textbox(label="Collection Name to Delete", placeholder="news_2024")
                delete_button = gr.Button("Delete")
        ragfile_out = gr.Textbox(label="Model Response", lines=6)

    mm_button.click(process_mm_request, inputs=mm_input, outputs=mm_out)
    rag_button.click(process_rag_request, inputs=[rag_input], outputs=rag_out)
    upl_button.click(process_rag_request, inputs=[rag_textfile_input, rag_file_input], outputs=ragfile_out)
    delete_button.click(delete_collection_request, inputs=[delete_file_name], outputs=ragfile_out)

# Launch the combined interface
demo.launch(server_name="0.0.0.0", auth=(get_all_usernames()))