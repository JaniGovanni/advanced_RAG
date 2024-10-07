# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies for unstructured
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    libreoffice \
    pandoc \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r clean_requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variables
ENV CHROMA_PATH=/data/chroma
ENV SOURCE_TO_ID_PATH=/data/chroma/source_to_ids.json
ENV MEMORY_STORAGE_PATH=/data/messages/
ENV UPLOADED_FILES_PATH=/data/uploaded_files
# http://host.docker.internal is the ip, which the container can use to access the host machine
# 11434 is the port, which ollama is running on the host machine
# http://localhost:11434 would not work, because it would refer to the container itself not host
# 
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# Create directories for persistent data
RUN mkdir -p /data

# Run app.py when the container launches
CMD ["streamlit", "run", "main.py"]

# Build the Docker image
# docker login
# docker build -t your-app-name .

# Run the Docker container
# for volume: you can use advanced_rag_storage:/data
# this will create a volume advanced_rag_storage in the docker default
# path for the volume (/var/lib/docker/volumes/advanced_rag_storage/data
# docker run -p 8501:8501 -e GROQ_API_KEY=your_groq_api_key -v /Users/jan/Desktop/stuff:/data your-app-name