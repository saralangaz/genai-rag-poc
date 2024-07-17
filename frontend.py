import gradio as gr
import requests
from constants import mm_models, rag_models, rag_input_choices
import os
import logging
from dotenv import load_dotenv
from utils import get_all_usernames
load_dotenv()

# Load environment variables
backend_host = os.getenv('BACKEND_HOST', "http://backend:8000")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a function to process the request
def process_mm_request(model, use_case, request: gr.Request, system_prompt=None, user_prompt=None, image_url=None):
    url = f"{backend_host}/api/process"  # Adjust this to your FastAPI server address
        
    payload = {
        "model": model,
        "use_case": use_case,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "image_url": image_url,
        "username": request.username,
        "input_choice": None
    }
    logger.info(f'Info loaded: Payload is {payload}')
    
    # Send POST request to FastAPI backend
    response = requests.post(url, data=payload, files=None)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

def process_rag_request(model, use_case, request: gr.Request, user_prompt, input_choice, files=None):
        url = f"{backend_host}/api/process"  # Adjust this to your FastAPI server address
        
        payload = {
            "model": model,
            "use_case": use_case,
            "system_prompt": None,
            "user_prompt": user_prompt,
            "image_url": None,
            "username": request.username,
            "input_choice": input_choice
        }
        logger.info(f'Info loaded: Payload is {payload}')
        
        # Send POST request to FastAPI backend
        processed_files = []
        if files:
            for file in files:
                if file:
                    processed_files.append(('document_files', open(file.name, 'rb')))
        response = requests.post(url, data=payload, files=processed_files)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}

# Define a function to retrieve old requests
def retrieve_request(request_id):
    url = f"{backend_host}/api/retrieve/{request_id}"  # Use the Azure service name
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}
    
# Define a function to update the visibility of inputs based on dropdown selection
def update_input_components(choice):
    if choice == "Ask a question to the knowledge base":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    elif choice == "Upload a Document to the knowledge base":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    elif choice == "Run a similarity search and return the appropiate documents":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("Choose the Use Case for this demo.")
    # Define Gradio components for multi-modal m:odel processing
    with gr.Tab("Multi-Modal Use Case"):
        mm_model_input = gr.Dropdown(label="Model", choices=mm_models)
        mm_use_case = gr.Textbox(label="Use Case", value="multimodal", visible=False)
        mm_system_prompt_input = gr.Textbox(label="System Prompt", placeholder="You are an intelligent assistant that helps generate structured data.")
        mm_user_prompt_input = gr.Textbox(label="User Prompt", placeholder="Generate a set of JSON data for a hypothetical e-commerce website."
                                    "The data should include fields for product name, price, and availability." 
                                    "The output should only include the data. Don't include any other message.")
        mm_image_url_input = gr.Textbox(label="Image URL", placeholder="https://example.com/image.png")
        mm_button = gr.Button("Submit")
        mm_out = gr.JSON()
    
    with gr.Tab("Rag Use Case"):
        # Define Gradio components for RAG model processing
        rag_model_input = gr.Dropdown(label="Model", choices=rag_models)
        rag_use_case = gr.Textbox(label="Use Case", value="rag", visible=False)
        rag_input_choice = gr.Dropdown(label="Input Type", choices=rag_input_choices)
        rag_text_input = gr.Textbox(label="User Query", placeholder="Question: Can you summarize [topic]? / Similarity Search: Return the documents that talk about [topic]", visible=False)
        rag_file_input = gr.Files(label="Upload Documents", type="filepath", visible=False)
        rag_out = gr.JSON()
        rag_button = gr.Button("Submit")
        # Update visibility of text and file input based on dropdown selection
        rag_input_choice.change(update_input_components, inputs=[rag_input_choice], outputs=[rag_text_input, rag_file_input, rag_text_input])

    with gr.Tab("ID Request Retrieval"):
        # Define Gradio components for model retrieval
        request_id_input = gr.Textbox(label="Request ID to Retrieve", placeholder="550e8400-e29b-41d4-a716-446655440000")
        ret_button = gr.Button("Retrieve")
        ret_out = gr.JSON()
   
    mm_button.click(process_mm_request, inputs=[mm_model_input, mm_use_case, mm_system_prompt_input, mm_user_prompt_input, mm_image_url_input], outputs=mm_out)
    rag_button.click(process_rag_request, inputs=[rag_model_input, rag_use_case, rag_text_input, rag_input_choice, rag_file_input], outputs=rag_out)
    ret_button.click(retrieve_request, inputs=request_id_input, outputs=ret_out)

# Launch the combined interface
demo.launch(server_name="0.0.0.0", auth=(get_all_usernames()))