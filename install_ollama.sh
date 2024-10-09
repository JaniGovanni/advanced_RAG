#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install curl based on the OS
install_curl() {
    if command_exists apt-get; then
        sudo apt-get update && sudo apt-get install -y curl
    elif command_exists yum; then
        sudo yum install -y curl
    elif command_exists brew; then
        brew install curl
    else
        echo "Error: Unable to install curl. Please install it manually."
        exit 1
    fi
}

# Check if curl is installed, if not, try to install it
if ! command_exists curl; then
    echo "curl is not installed. Attempting to install..."
    install_curl
fi

# Install Ollama
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    brew install ollama
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Unsupported operating system"
    exit 1
fi

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start (increased wait time for slower systems)
sleep 15

# Pull the models
ollama pull llama3.2
ollama pull llama3.2:1b
ollama pull nomic-embed-text

echo "Ollama installation and model pulling completed successfully!"
