import gradio as gr
import requests
import numpy as np


# Define a function to process the request
def process_request(system_prompt, user_prompt, model, output_type, image_url=None):
    url = "http://backend:8000/api/process"  # Adjust this to your FastAPI server address
    
    # Prepare payload for text data
    payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model": model,
        "output_type": output_type,
        "image_url": image_url
    }

    # Send POST request to FastAPI backend
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# Define a function to retrieve old requests
def retrieve_request(request_id):
    url = f"http://backend:8000/api/retrieve/{request_id}"  # Use the Docker service name
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# Define Gradio components for text model processing
system_prompt_input = gr.Textbox(label="System Prompt", placeholder="You are an intelligent assistant that helps generate structured data.")
user_prompt_input = gr.Textbox(label="User Prompt", placeholder="Generate a set of JSON data for a hypothetical e-commerce website."
                               "The data should include fields for product name, price, and availability." 
                               "The output should only include the data. Don't include any other message.")
model_input = gr.Dropdown(label="Model", choices=["llama3", "mistral", "gemma2", "llava:13b"])
output_type_input = gr.Textbox(label="Output Type", placeholder="JSON array")
image_url_input = gr.Textbox(label="Image URL. Use only with llava model", placeholder="https://example.com/image.png")

# Define Gradio components for model retrieval
request_id_input = gr.Textbox(label="Request ID to Retrieve", placeholder="550e8400-e29b-41d4-a716-446655440000")

# Create Gradio interface for model processing
process_interface = gr.Interface(
    fn=process_request,
    inputs=[system_prompt_input, user_prompt_input, model_input, output_type_input, image_url_input],
    outputs="json",
    title="Ollama New Model Request",
    description="Hello and welcome to our amazing app! To use the model, enter the following info."
)
#
#  Create Gradio interface for retrieval
retrieve_interface = gr.Interface(
    fn=retrieve_request,
    inputs=request_id_input,
    outputs=gr.JSON(),
    title="Ollama Retrieve Past Request",
    description="Retrieve a past request using its request ID"
)

# Combine interfaces
combined_interface = gr.TabbedInterface(
    [process_interface, retrieve_interface],
    ["Process new model request", "Retrieve old model response"]
)

# Launch the combined interface
combined_interface.launch(server_name="0.0.0.0")