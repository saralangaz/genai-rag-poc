from pydantic import BaseModel
from typing import ClassVar, List

# Generic system prompt
gen_system_prompt = "You are a helpful AI assistant"
# Generic user prompt for text
gen_text_prompt = "Summarize these documents"
# Generic user prompt for image
gen_image_prompt = "Describe this image"
# Generic user query
gen_user_query = ""
# Generic context
gen_context = ""

# Create Pydantic classes
class ExecuteModelInput(BaseModel):
    model: str
    use_case: str
    collection_name: str | None
    k_value: int | None
    system_prompt: str | None
    user_prompt: str | None
    image_url: str | None
    username: str

class CollectionInput(BaseModel):
    collection_name: str

class UploadDocuments(BaseModel):
    model: str
    collection_name: str

class ValidModels(BaseModel):
    models: ClassVar[List[str]] = ["llama2:7b", "llama3.1:8b", "llava:7b", "mistral:7b"]