import streamlit as st
import requests
from dotenv import load_dotenv
import os

#load_dotenv('app/.env')

BASE_URL = os.getenv('BASE_URL')

def get_default_lists():
    response = requests.get(f"{BASE_URL}/get_default_lists")
    if response.status_code == 200:
        data = response.json()
        return data.get('default_lists', {})
    else:
        st.error(f"Failed to fetch default lists. Status code: {response.status_code}")
        return {}

def api_process_doc(config):
    url = f"{BASE_URL}/process_doc"
    response = requests.post(url, json=config.model_dump())
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.json()['message']}")
        return None

def api_get_result(query, config):
    url = f"{BASE_URL}/get_result"
    payload = {
        "query": query,
        "config": config
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.json()['message']}")
        return None

def get_stored_tags_and_files():
    response = requests.get(f"{BASE_URL}/get_stored_tags_and_files")
    if response.status_code == 200:
        data = response.json()
        return data.get('tag_to_files', {})
    else:
        st.error(f"Failed to fetch stored tags and files. Status code: {response.status_code}")
        return {}

def upload_file(file):
    """
    Uploads the file to the Flask server
    """
    files = {'file': (file.name, file.getvalue())}
    try:
        response = requests.post(f"{BASE_URL}/upload_file", files=files)
        response.raise_for_status()
        
        data = response.json()
        if response.status_code == 200:
            st.success('File uploaded successfully.')
            return data.get('file_id')
        else:
            st.error(f'Error uploading file: {data.get("message", "Unknown error")}')
    
    except requests.exceptions.RequestException as e:
        st.error(f'Error communicating with server: {str(e)}')
    except ValueError as e:
        st.error(f'Error parsing server response: {str(e)}')
    except Exception as e:
        st.error(f'Unexpected error: {str(e)}')
    
    return None