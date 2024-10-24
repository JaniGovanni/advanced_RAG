#!/bin/bash

# run: ./startup.sh

# Start Flask API
echo "Starting Flask API..."
python app/api_setup/api.py &

# Wait for Flask to start
sleep 2

# Start Streamlit
echo "Starting Streamlit..."
streamlit run main.py

# Wait for both processes
wait