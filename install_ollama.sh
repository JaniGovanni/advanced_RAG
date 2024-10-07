#!/bin/bash

curl -fsSL https://ollama.com/install.sh | sh
# Start Ollama in the background
ollama serve &

# Wait for Ollama to start
sleep 10

# Pull the models
ollama pull llama3.2
ollama pull nomic-embed-text
