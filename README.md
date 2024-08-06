# FastAPI and Gradio Application

Hello and welcome!

This repository contains a FastAPI backend and a Gradio frontend for processing and retrieving data from the Ollama model. The application can handle both text and image inputs, using different models from Ollama. The embeddings are stored in Weaviate vectorial DB.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Endpoints](#endpoints)
- [Deployment on Azure](#deployment-on-azure)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Requirements

- Docker
- Docker Compose
- Python 3.10 or higher

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/saralangaz/ollama-poc.git
    cd ollama-poc
    ```

2. Build the Docker images:
    ```sh
    docker-compose build
    ```

## Running the Application

1. Start the application using Docker Compose:
    ```sh
    docker-compose up
    ```

2. The application will be available at the following URLs:
    - Gradio Frontend: `http://localhost:7860`
    - FastAPI Backend: `http://localhost:8000`
    - Ollama container: `http://localhost:11434`
    - Weaviate container: `http://localhost:8080`

## Endpoints

### FastAPI

- **POST /api/process**: Processes the input text and/or image using the specified Ollama model.
    - **Request Parameters**:
        - `system_prompt` (string): The system prompt for the model.
        - `user_prompt` (string): The user prompt for the model.
        - `model` (string): The model to be used (e.g., `llava`).
        - `output_type` (string): The expected output type (e.g., `JSON array`).
        - `image_url` (optional, string): The image url to be processed.
    - **Response**:
        - `id` (string): The request ID.
        - `data` (string): The processed data.

- **GET /api/retrieve**: Retrieves the processed data using the request ID. (DEPRECATED)
    - **Request Parameters**:
        - `request_id` (string): The ID of the request to retrieve.
    - **Response**:
        - `data` (string): The processed data.

## Deployment on Azure

This application has been deployed inside a Kubernetes cluster. All the required yaml files are stored inside the kubernetes folder.

# Usage

1. Open the Gradio interface at http://localhost:7860 (or your Azure Web App URL).
   - This script creates a multi-tab interface that allows you to choose between different use cases:
     - **Multi-Modal Use Case**: Select a model and provide system prompts, user prompts, and optional image URLs to generate structured data.
     - **RAG Use Case**: Choose a model and input type (question or similarity search), and optionally upload documents for retrieval.
     - **ID Request Retrieval**: Enter a request ID to retrieve previously processed data.

2. Fill in the Required Fields:
   - **Multi-Modal Use Case**:
     - Choose a model from the dropdown list.
     - Enter system and user prompts as text inputs.
     - Optionally provide an image URL.
     - Click `Submit` to process the request.

   - **RAG Use Case**:
     - Select a model from the dropdown list.
     - Choose the type of input (question or similarity search).
     - Enter the user query or upload documents using the provided file upload button.
     - Click `Submit` to process the request.


3. Review the Output:
   - The interface will display the processed data in JSON format.
   - For multi-modal and RAG use cases, the processed data may include structured outputs or retrieved documents based on the input provided.


## Model Classes

This repository includes several model classes implemented in `models.py` that facilitate different functionalities:

### MultiModalModel

- **Purpose**: Integrates text and image inputs for processing using the Ollama model.
- **Methods**:
  - `execute_model(file_path=None, document_file=None)`: Processes input text and optionally an image, using system and user prompts.

### RagModel

- **Purpose**: Implements the RAG (Retrieval-Augmented Generation) model for document handling and retrieval.
- **Methods**:
  - `execute_model(file_paths, document_files)`: Handles document processing, indexing, and retrieval based on user queries.

Each model class provides specific functionalities tailored to different use cases, enhancing the capabilities of the FastAPI and Gradio application in handling various types of data inputs and requests.


## Troubleshooting

- **Connection errors**:
  Verify the backend service URL and ensure the Docker containers are running and accessible.

- **FileNotFoundError**:
  Ensure the file paths are correct and the uploaded files are being handled appropriately.

- **`AttributeError: 'numpy.ndarray' object has no attribute 'read'`**:
  Ensure the image is being processed correctly before being sent to the backend.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio](https://gradio.app/)
- [Ollama](https://ollama.com/)
- [Weaviate](https://weaviate.com/)
