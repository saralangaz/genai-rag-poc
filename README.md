# FastAPI and Gradio Application

Hello and welcome!

This repository contains a FastAPI backend and a Gradio frontend for processing and retrieving data from the Ollama model. The application can handle both text and image inputs, using the LLava model from Ollama.

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

- **GET /api/retrieve**: Retrieves the processed data using the request ID.
    - **Request Parameters**:
        - `request_id` (string): The ID of the request to retrieve.
    - **Response**:
        - `data` (string): The processed data.

## Deployment on Azure

1. Push the Docker images to your Azure Container Registry (ACR):
    ```sh
    docker tag yourimage:tag acrname.azurecr.io/yourimage:tag
    docker push acrname.azurecr.io/yourimage:tag
    ```

2. Create an Azure Web App for Containers and configure it to use your images from ACR.

3. Set up the following environment variables in your Azure Web App:
    - `WEBSITES_PORT` to `80` for the Gradio container.

4. Ensure your Azure Web App can access the frontend service.

## Usage

1. Open the Gradio interface at `http://localhost:7860` (or your Azure Web App URL).

2. Fill in the required fields:
    - `System Prompt`: Enter the system prompt for the model.
    - `User Prompt`: Enter the user prompt for the model.
    - `Model`: Select the model from the dropdown list.
    - `Output Type`: Specify the expected output type.
    - `Image URL`: Specify an image URL if needed.

3. Click `Submit` to process the request.

4. Use the request ID provided in the response to retrieve an past request.

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
