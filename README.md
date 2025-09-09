# End-to-End MLOps: A Production-Ready Customer Churn Prediction API


![alt text](https://github.com/JUBRIL-ADEBOGUN/Telco-custtomer-churn/actions/workflows/main.yml/badge.svg)]


## Project Overview

This project goes beyond a typical data science notebook. It is a complete, end-to-end implementation of a machine learning system designed to predict customer churn. The primary goal was not just to build an accurate model, but to engineer a robust, scalable, and automated MLOps pipeline for serving it as a production-ready API.

The system automatically tests, packages, and prepares the application for deployment whenever new code is pushed to the main branch, ensuring reliability and a streamlined development lifecycle.
Live API Endpoint: [Add your deployed Google Cloud Run URL here]


Core Features

Machine Learning Model: A Random Forest Classifier trained on customer data to predict churn with high accuracy.
RESTful API: A high-performance API built with FastAPI to serve model predictions over the web. Includes automatic data validation and interactive documentation.
Containerization: The entire application and its dependencies are packaged into a lightweight, portable Docker container for consistent execution in any environment.
Continuous Integration (CI): Every code push automatically triggers a workflow that installs dependencies and runs a full suite of Pytest unit tests to ensure code quality and prevent regressions.
Continuous Delivery (CD): Upon successful testing, the CI/CD pipeline automatically builds a new Docker image and pushes it to a container registry, making it ready for deployment.
Cloud Deployment: The containerized application is deployed on Google Cloud Run, a serverless platform that automatically scales to handle traffic—from zero to thousands of requests—ensuring high availability and cost-efficiency.
Architecture & Tech Stack
The project follows a modern MLOps workflow, leveraging a suite of powerful tools to automate the path from code to production.
Workflow Diagram:
Code Push (GitHub) -> CI/CD Trigger (GitHub Actions) -> Test (Pytest) -> Build Image (Docker) -> Push to Registry -> Deploy (Google Cloud Run)
Key Technologies:
ML & Data Science: Scikit-learn, Pandas, Joblib
API & Backend: FastAPI, Uvicorn
MLOps & DevOps: Docker, GitHub Actions, Pytest
Cloud Platform: Google Cloud Run, Google Artifact Registry
Getting Started
You can run this project locally for development and testing.
Prerequisites
Python 3.9+
Docker Desktop
Git
Local Installation & Setup
Clone the repository:
code
Bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
Create and activate a virtual environment:
code
Bash
python -m venv .venv
source .venv/bin/activate
Install dependencies:
code
Bash
pip install -r requirements.txt
Run the tests:
code
Bash
pytest
Running the API
1. Via Uvicorn (for development)
This will run the API server in development mode with hot-reloading.
code
Bash
uvicorn main:app --reload
The API will be available at http://127.0.0.1:8000. Access the interactive documentation at http://127.0.0.1:8000/docs.
2. Via Docker (to replicate the production environment)
This builds and runs the containerized application.
code
Bash
# Build the Docker image
docker build -t churn-api .

# Run the container
docker run -p 8080:80 churn-api
The API will be available at http://localhost:8080. Access the interactive documentation at http://localhost:8080/docs.
API Endpoints
POST /predict
Receives customer data and returns a churn prediction.
Request Body:
code
JSON
{
  "Tenure": 12,
  "MonthlyCharges": 59.99,
  "Gender": 1
}
Note: Gender is pre-encoded (e.g., Male: 1, Female: 0).
Success Response (200 OK):
code
JSON
{
  "prediction": 0,
  "prediction_label": "No Churn",
  "class_probabilities": {
    "No Churn": 0.85,
    "Churn": 0.15
  }
}
Future Improvements
Model Monitoring: Implement a system to monitor the live model for performance degradation or data drift.
Automated Retraining: Create a separate workflow to automatically retrain the model on new data and register the new version.
Experiment Tracking: Integrate a tool like MLflow or Weights & Biases to log experiments, track model metrics, and manage model versions systematically.
