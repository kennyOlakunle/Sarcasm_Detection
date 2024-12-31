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
### Prerequisites
**Python 3.8+** recommended (works with 3.9, 3.10, etc.).

**Virtual Environment** (e.g., venv or Conda) to keep dependencies isolated.

**Git** for cloning and version control.

### Installation

1. **Clone the repository** (or download the ZIP):

```
git clone https://github.com/<YOUR_USERNAME>/sarcasm_detection_app.git
cd sarcasm_detection_app
```

2. **Create and activate a virtual environment:**
```
# Using venv example
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

3. **Install dependencies:**
```
pip install --upgrade pip
pip install -r requirements.txt

```

### Usage

1. **Dataset:** Make sure you have your sarcasm dataset (e.g., `sarcasm_dataset.csv`) in the `data/` folder.
2. **Model:** You can train the model yourself or download a pre-trained model and place it in the `models/` folder.
3. **Run App:** Once everything is set up, proceed to Running Locally.

### Model Training
#### Key Steps:
1. Load data from `data/sarcasm_dataset.csv`.
2. Perform EDA (optional but recommended) to visualize label distribution and outliers.
3. Train the model using LSTM for 2-class classification (sarcastic vs. not sarcastic).
4. Save the trained model and tokenizer to the directory.

**Note:** Ensure the saved model folder includes all necessary files.

### Streamlit Application
The `app.py` file houses the **Streamlit** application. It:

- Loads the trained model and tokenizer using `@st.cache_resource`.
- Provides a Text Area for user input.
- Runs Inference and outputs a sarcastic or non-sarcastic label.

### Running Locally
1. Activate your virtual environment (if not already).
2. Navigate to the project root:
```
cd path/to/sarcasm_detection_app
```
3. Launch the Streamlit app:
```
streamlit run app.py
```

4. Interact with the local server URL displayed in your terminal (e.g., `http://localhost:8501`).


### Deploying to Streamlit Cloud
1. Push your code to a GitHub repository.
2. Sign in to [Streamlit Cloud](https:://streamlit.io/).
3. Create a New App and connect your GitHub repository.
4. Specify app.py under “File path” and confirm the branch you want to deploy from.
5. Deploy. After a few moments, Streamlit Cloud will build and run your app, providing a shareable URL.

