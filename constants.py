from pydantic import BaseModel

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

# Multi-modal models
mm_models = ["llama2:7b", "llama3.1:8b", "llava:7b"]
# Rag models
rag_models = ["llama2:7b", "llama3.1:8b", "mistral:7b"]

# Create Pydantic classes
class InputText(BaseModel):
    model: str | None
    use_case: str
    collection_name: str | None
    system_prompt: str | None
    user_prompt: str | None
    image_url: str | None
    username: str