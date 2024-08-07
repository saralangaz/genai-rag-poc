import gradio as gr
import requests
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
        url = f"{backend_host}/api/execute_model"  
        input = json.loads(input)
        
        payload = {
            "model": input.get("model"),
            "use_case": "multimodal",
            "collection_name": None,
            "k_value": None,
            "system_prompt": input.get("system_prompt"),
            "user_prompt": input.get("user_prompt"),
            "image_url": input.get("image_url"),
            "username": request.username
        }
        logger.info(f'Info loaded: Payload is {payload}')
        
        # Send POST request to FastAPI backend
        response = requests.post(url, json=payload, stream=True)
        collected_data = ""
        for line in response.iter_lines():
            decoded_line = line.decode('utf-8')
            if decoded_line != None:
                collected_data += decoded_line
                yield collected_data
    except Exception as e:
        return {'error': f'{str(e)}'}
    
def process_rag_request(input: dict, request: gr.Request):
    """
    Send a POST request to a FastAPI backend for processing RAG (Retrieval-Augmented Generation) requests.

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
        url = f"{backend_host}/api/execute_model" 
        input = json.loads(input)

        payload = {
            "model": input.get("model"),
            "use_case": "rag",
            "collection_name": input.get("collection_name"),
            "k_value": input.get("k"),
            "system_prompt": None,
            "user_prompt": input.get("user_prompt"),
            "image_url": None,
            "username": request.username
        }

        logger.info(f'Info loaded: Payload is {payload}')
        # Send POST request to FastAPI backend
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    
    except Exception as e:
        return {'error': f'{str(e)}'}

def delete_collection_request(input: str):
    """
    Send a POST request to a FastAPI backend for deleting a Collection from a Vectorial DB.

    Parameters:
    -----------
    input : str
        The collection Name to be deleted.

    Returns:
    --------
    dict
        JSON response from the FastAPI backend.
    """
        
    try:
        url = f"{backend_host}/api/delete_collection" 

        payload = {
            "collection_name": input
        }
        logger.info(f'Info loaded: Payload is {payload}')
        
        # Send POST request to FastAPI backend
        response = requests.post(url, json=payload)
        return response.text
    
    except Exception as e:
        return {'error': f'{str(e)}'}

def list_collection_request():
    """
    Send a GET request to a FastAPI backend for listing all collections from Vectorial DB.

    Returns:
    --------
    dict
        JSON response from the FastAPI backend.
    """
        
    try:
        url = f"{backend_host}/api/list_collections" 
        
        # Send GET request to FastAPI backend
        response = requests.get(url)
        
        return response.text
    
    except Exception as e:
        return {'error': f'{str(e)}'}

def upload_documents(input:str, request:gr.Request, files:list):
    """
    Send a POST request to a FastAPI backend for uploading documents to a Vectorial DB.

    Parameters:
    -----------
    input : str
        The collection Name to be deleted.
    request: gr.Request
        The username.
    files : List[UploadFile]
        List of files to be uploaded for the request.

    Returns:
    --------
    dict
        JSON response from the FastAPI backend.
    """
    url = url = f"{backend_host}/api/load_documents" 
    processed_files = []
    for file in files:
        if file:
            processed_files.append(('document_files', open(file.name, 'rb')))
    payload = {
        "collection_name": input,
        "username": request.username
        }
    response = requests.post(url, data=payload, files=processed_files)
    
    return response.text

css = """
.upload-section {
    width: 100%;
    max-width: 600px; /* Adjust the max-width as needed */
}
"""

# Create Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("""Choose the Use Case for this demo.
                
                **NEXT STEPS y MEJORAS**
                - Mejorar tiempos de respuesta de Ollama API
                - Mejorar chunks de textos
                - Posibilidad de filtrar queries a la BBDD
                - Capacidad de cargar dinámicamente nuevos modelos en el contenedor de Ollama
                - Cargar imágenes en el Multi-Modal Use Case (en vez de pasar una URL)
                - Asociar collections a usuarios en la BBDD""")
    # Define Gradio components for multi-modal model processing
    with gr.Tab("Ask the chatbot"):
        gr.Markdown("""**Multi-Modal Use Case:** 
                    This tab is used to ask the chatbot to generate responses based on your queries, provide information about an image, etc.
                    
                    Mandatory inputs: model type (llama3.1:8b, llava:7b). 
                    
                    Optional inputs: system prompt, user prompt, image URL. "

                    USE LLAVA MODEL FOR IMAGE PROCESSING!

                    **Structure**:""")
        gr.Markdown("""```
                    {"model": "llama3.1:8b",
                     "system_prompt": "You are an intelligent assistant that helps generate structured data.",
                     "user_prompt":"Generate a set of JSON data for a hypothetical e-commerce website. The data should include fields for product name, price, and availability",
                     "image_url": ""}
                ```""")
        mm_input = gr.Textbox(label="Inputs")
        mm_button = gr.Button("Submit")
        mm_out = gr.Textbox(label="Model Response", lines=10)
    
    with gr.Tab("Ask the Knowledge Base"):
        # Define Gradio components for RAG model processing
        gr.Markdown("""**Rag Use Case:** 
                    This tab is used to Upload documents to Weaviate Vectorial Database and ask questions or extract information from those documents.
                    
                    Mandatory inputs: model type (llama3.1:8b, mistral:7b), collection name and user prompt. 
                    
                    Optional input: k: number of documents to retrieve, by default 5;
                    
                    Embeddings will be stored in weaviate under the collection name.
                    You can save all the documents you want inside the same collection name.""")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Structure")
                gr.Markdown("""```
                {"model": "llama3.1:8b",
                "collection_name": "kubernetes",
                "k": 5,
                "user_prompt": "can you summarize the use of kubernetes in 100 words"}
                ```""")
                rag_input = gr.Textbox(label="Inputs", lines=4)
        rag_button = gr.Button("Submit")
        rag_out = gr.Textbox(label="Model Response", lines=6)

    with gr.Tab("Handle Vectorial Database"):
        with gr.Row():
            with gr.Column(scale=1,):
                gr.Markdown("### Upload Documents to Weaviate DB")
                gr.Markdown("Add a collection name and select one or more documents to upload and save them in the DB")
                rag_textfile_input = gr.Textbox(label="Collection Name", placeholder="kubernetes")
                rag_file_input = gr.Files(label="Upload Documents", type="filepath")
                upl_button = gr.Button("Upload")
            
            with gr.Column(scale=1):
                gr.Markdown("### Delete a Collection from Weaviate DB")
                delete_file_name = gr.Textbox(label="Collection Name", placeholder="kubernetes")
                delete_button = gr.Button("Delete")
                gr.Markdown("### List all collections from Weaviate DB")
                list_button = gr.Button("List")
        ragfile_out = gr.Textbox(label="Database Response")

    mm_button.click(process_mm_request, inputs=mm_input, outputs=mm_out)
    rag_button.click(process_rag_request, inputs=[rag_input], outputs=rag_out)
    upl_button.click(upload_documents, inputs=[rag_textfile_input, rag_file_input], outputs=ragfile_out)
    delete_button.click(delete_collection_request, inputs=[delete_file_name], outputs=ragfile_out)
    list_button.click(list_collection_request, outputs=ragfile_out)

# Launch the combined interface
demo.launch(server_name="0.0.0.0", auth=(get_all_usernames()))
