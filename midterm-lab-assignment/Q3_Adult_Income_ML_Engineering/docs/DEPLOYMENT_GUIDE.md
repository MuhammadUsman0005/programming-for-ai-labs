# Q3 deployment guide — Hugging Face Spaces (Docker)

This project includes a Dockerfile that starts FastAPI on an internal local port and Streamlit on the Space's public application port. This keeps the Streamlit interface talking to the FastAPI service inside the same container.

## Before deploying

1. Download the Adult dataset and run the full training command locally:
   `python -m src.train_adult --automl`
2. Confirm `models/model_bundle.joblib` exists.
3. Run FastAPI and Streamlit locally and test a prediction.
4. Do not upload the dataset CSV if it is not needed for inference. The saved model bundle is the deployed inference artifact.

## Create a Space

1. Sign in to Hugging Face.
2. Create a new Space.
3. Choose **Docker** as the SDK/runtime.
4. Choose a public or private visibility according to your assignment needs.
5. Clone the Space's Git repository using the instructions shown by Hugging Face.
6. Copy the Q3 project files into the Space repository. Include `models/model_bundle.joblib`, the `api/`, `src/`, and application files, `Dockerfile`, and `requirements_deploy.txt`.
7. Commit and push the files to the Space repository.
8. Wait for the Space build to finish. Open the Space URL and test the form.
9. If a prediction fails, inspect build/runtime logs and confirm the model bundle was uploaded and the model dependencies match the training environment.

## Docker Space configuration

The project-root `README.md` begins with the required Space metadata:

```yaml
---
title: Adult Income ML Engineering Demo
sdk: docker
app_port: 7860
---
```

## Deploy through Git

The exact clone command/token varies by your account and Hugging Face's current UI. Use the command provided by your Space page rather than copying a token into a source file. Never put access tokens in GitHub or the Space repository.

## Streamlit Community Cloud alternative

If you choose Streamlit Community Cloud instead, deploy `streamlit_app.py` as the app. You must first host FastAPI somewhere publicly reachable and set the Streamlit secret/environment variable `API_URL` to that public API base URL. The default `127.0.0.1:8000` only works locally or inside the included Docker deployment; it cannot be used by a cloud-hosted Streamlit app to reach your personal computer.

## Final report

Copy the real public Space/app URL into the Q3 report only after a clean browser test. A deployment guide and Dockerfile are included, but this starter package has not been published under your personal account and therefore has no live link yet.
