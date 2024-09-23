# advanced_rag

for unstructured you have to install sudo apt install libmagic1 poppler-utils libreoffice pandoc tesseract-ocr

also you need to install ollama with curl -fsSL https://ollama.com/install.sh | sh

and pull the models phi3.5 and nomic-embed-text with ollama

on mac and windows it is better to run docker outside a container, because the docker
there is a performance decrease, because the docker container must also 
run a linux virtual machine to run the ollama server. On linux this is not necessary.
Also on mac, there seems to be no way for a container to access the GPU
