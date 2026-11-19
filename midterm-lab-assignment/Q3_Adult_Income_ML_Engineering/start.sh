#!/usr/bin/env bash
set -e

# Start the API inside the container. Streamlit calls it through localhost.
uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# Hugging Face Spaces exposes the Streamlit port.
exec streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 7860 --server.headless true
