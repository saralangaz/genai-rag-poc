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
mm_models = ["llama3", "mistral",  "llava:13b", "gemma2"]
# Rag models
rag_models = ["mistral:7b-instruct", "gemma:7b", "llama3", "mistral",]
rag_input_choices = ["Upload a Document to the knowledge base", "Ask a question to the knowledge base", "Run a similarity search and return the appropiate documents"]

# Create Pydantic classes
class InputText(BaseModel):
    model: str
    use_case: str
    system_prompt: str | None
    user_prompt: str | None
    image_url: str | None
    username: str
    input_choice: str | None