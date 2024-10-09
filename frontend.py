import gradio as gr
import requests
import os
import logging
from dotenv import load_dotenv
from utils import get_all_usernames
import json
import re
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackendAPI:
    def __init__(self):
        self.backend_host = os.getenv('BACKEND_HOST', "http://backend:8000")

    def process_rag_request(self, input: dict, request: gr.Request):
        """
        Send a POST request to a FastAPI backend for processing RAG (Retrieval-Augmented Generation) requests.
        """
        try:
            url = f"{self.backend_host}/api/execute_model"
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
            response = requests.post(url, json=payload)
            formatted_response = re.sub(r"\\n\\n", "</p><p>", response.text)
            formatted_response = re.sub(r"\\n", "<br>", formatted_response) 

            return f"<p>{formatted_response}</p>" 

        except Exception as e:
            return {'error': f'{str(e)}'}

    def delete_collection_request(self, collection_name: str):
        """
        Send a POST request to a FastAPI backend for deleting a Collection from a Vectorial DB.
        """
        try:
            url = f"{self.backend_host}/api/delete_collection"
            payload = {"collection_name": collection_name}
            logger.info(f'Info loaded: Payload is {payload}')
            response = requests.post(url, json=payload)
            return response.text
        except Exception as e:
            return {'error': f'{str(e)}'}

    def list_collection_request(self):
        """
        Send a GET request to a FastAPI backend for listing all collections from Vectorial DB.
        """
        try:
            url = f"{self.backend_host}/api/list_collections"
            response = requests.get(url)
            return response.text
        except Exception as e:
            return {'error': f'{str(e)}'}

    def upload_documents(self, collection_name: str, request: gr.Request, files: list):
        """
        Send a POST request to a FastAPI backend for uploading documents to a Vectorial DB.
        """
        url = f"{self.backend_host}/api/load_documents"
        processed_files = []
        for file in files:
            if file:
                processed_files.append(('document_files', open(file.name, 'rb')))
        payload = {
            "collection_name": collection_name,
            "username": request.username
        }
        response = requests.post(url, data=payload, files=processed_files)
        return response.text

    def upload_images(self, collection_name: str, image_description:str, request: gr.Request, file: str):
        """
        Send a POST request to a FastAPI backend for uploading images to a Vectorial DB.
        """
        url = f"{self.backend_host}/api/load_images"  
        processed_file = [('image_files', open(file, 'rb'))]

        payload = {
            "collection_name": collection_name,
            "image_description": image_description,
            "username": request.username
        }

        response = requests.post(url, data=payload, files=processed_file)
        # Close the opened files to avoid resource leaks
        for _, file in processed_file:
            file.close()
        return response.text

css = """
.upload-section {
    width: 100%;
    max-width: 600px; /* Adjust the max-width as needed */
}
"""

# Create an instance of the BackendAPI class
backend_api = BackendAPI()

# Load your header image
header_image = "/code/images/image.png"

# Create Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Image(header_image, elem_id="header-img", show_label=False,  # Hide the label
        show_download_button=False,  # Hide the download button
        show_fullscreen_button=False,
        width="100%") 
    gr.Markdown("""
                **NEXT STEPS y MEJORAS**
                - Mejorar tiempos de respuesta de Ollama API
                - Mejorar chunks de textos
                - Filtrado dinámico de queries a la BBDD
                - Capacidad de cargar dinámicamente nuevos modelos en el contenedor de Ollama
                - Asociar collections a usuarios en la BBDD
                """)
    
    with gr.Tab("Ask the Knowledge Base"):
        # Define Gradio components for RAG model processing
        gr.Markdown("""This tab is used to ask questions or extract information from the Knowledge Base.
                    
                    Mandatory inputs: model type (llama3.1:8b, mistral:7b), collection name and user prompt. 

                    The model type for image retrieval is llava:7b by default.
                    
                    Optional input: k: number of documents/images to retrieve, by default 5;
                    
                    Embeddings will be stored in WeaviateDB under the collection name.
                    You can save all the documents and images you want inside the same collection name.""")
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
        rag_out = gr.Markdown(label="Model Response", line_breaks=True, show_copy_button=True)

    with gr.Tab("Handle Knowledge Base"):
        with gr.Row():
            with gr.Column(scale=1,):
                gr.Markdown("### Upload Documents to Weaviate DB")
                rag_textfile_input = gr.Textbox(label="Collection Name", placeholder="kubernetes")
                rag_file_input = gr.Files(label="Upload Documents", type="filepath")
                upl_button = gr.Button("Upload")
            
            with gr.Column(scale=1,):
                gr.Markdown("### Upload Images to Weaviate DB")
                rag_textimage_input = gr.Textbox(label="Collection Name", placeholder="kubernetes")
                rag_textimagedesc_input = gr.Textbox(label="Image description", placeholder="kubernetes logo")
                image_input = gr.Image(type="filepath", label="Upload Image(s)")
                upl_img_button = gr.Button("Upload")
            
            with gr.Column(scale=1):
                gr.Markdown("### Delete a Collection from Weaviate DB")
                delete_file_name = gr.Textbox(label="Collection Name", placeholder="kubernetes")
                delete_button = gr.Button("Delete")
                gr.Markdown("### List all collections from Weaviate DB")
                list_button = gr.Button("List")
        ragfile_out = gr.Textbox(label="Database Response")

    rag_button.click(backend_api.process_rag_request, inputs=[rag_input], outputs=rag_out)
    upl_button.click(backend_api.upload_documents, inputs=[rag_textfile_input, rag_file_input], outputs=ragfile_out)
    upl_img_button.click(backend_api.upload_images, inputs=[rag_textimage_input, rag_textimagedesc_input, image_input], outputs=ragfile_out)
    delete_button.click(backend_api.delete_collection_request, inputs=[delete_file_name], outputs=ragfile_out)
    list_button.click(backend_api.list_collection_request, outputs=ragfile_out)

# Launch the combined interface
demo.launch(server_name="0.0.0.0", auth=(get_all_usernames()), allowed_paths=["/code/images/"])
