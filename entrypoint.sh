#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve LLAMA2 model..."
ollama pull llama2:7b
echo "🟢 Done!"
echo "🔴 Retrieve LLAMA3 model..."
ollama pull llama3.1:8b
echo "🟢 Done!"
echo "🔴 Retrieve LLaVa model..."
ollama pull llava:7b
echo "🟢 Done!"
echo "🔴 Retrieve Mistral model..."
ollama pull mistral:7b
echo "🟢 Done!"
echo "🔴 Retrieve all minilm model..."
ollama pull all-minilm
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid