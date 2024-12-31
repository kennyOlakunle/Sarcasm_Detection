# Sarcasm Detection App

A machine learning application built with Python, TensorFlow, Scikit-learn, NLTK, and Streamlit to detect whether a given text input is sarcastic or not. This project uses an open-source sarcasm-labeled dataset to build the model, then provides an intuitive web interface that allows users to input text and instantly receive a prediction.

### Table of Contents

1. Overview
2. Features
3. Project Structure
4. Getting Started
    - Prerequisites
    - Installation
    - Usage
5. Model Training
6. Streamlit Application
    - Running Locally
    - Deploying to Streamlit Cloud
7. Potential Enhancements
8. References
9. License


### Overview

Sarcasm can drastically change the tone or sentiment of a message. Whether you’re analyzing social media comments, moderating online forums, or developing chatbot interactions, accurately detecting sarcasm is crucial. This application demonstrates how to:

- Gather and preprocess a sarcasm dataset
- Train an LSTM model for sarcasm detection
- Serve real-time predictions through a user-friendly Streamlit interface

### Features

- **End-to-End Pipeline:** From data loading to inference, everything is streamlined in one repository.
- **Interactive Web App:** Leverages Streamlit to allow users to type (or paste) text and see immediate predictions.
- **LSTM model:** Build an LSTM model to handle context-sensitive sarcasm clues.
- **Expandable:** Easily adapt to other text classification tasks by changing the dataset or fine-tuning strategy.


### Getting Started
#### Prerequisites
**Python 3.8+** recommended (works with 3.9, 3.10, etc.).
**Virtual Environment** (e.g., venv or Conda) to keep dependencies isolated.
**Git** for cloning and version control.

#### Installation

