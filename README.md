# Mental Health Chatbot

## Overview

The Mental Health Chatbot is an AI-powered chatbot designed to provide mental health support and resources. Leveraging a knowledge base and a custom model from Hugging Face stored in a local machine, the chatbot aims to offer empathetic conversations, provide information on various mental health topics, and suggest coping strategies. This project seeks to create a safe and supportive environment for users seeking mental health assistance.

## Features

- **Empathetic Conversations**: The chatbot engages users with empathetic responses, helping them feel heard and supported.
- **Knowledge Base Integration**: Access to a rich knowledge base provides users with accurate information on mental health topics.
- **Custom Model**: Utilizes a custom model from Hugging Face to understand and respond to user queries effectively.
- **Resource Suggestions**: Offers suggestions for coping strategies, exercises, and other resources based on user input.
- **Confidential and Secure**: Ensures user privacy and confidentiality in all interactions.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Clone the Repository

```
git clone https://github.com/PrathmeshChurhe/Mental_health_chatbot
cd Mental_health_chatbot
```

- ### Create by using Python venv
    - Create a Python venv: ```python -m <envname> .venv```
    - Activate the Python venv: ```.venv/Scripts/activate.bat```
- ### Create by using conda env - python 3.9
    - Create a Python venv: ```conda create -n <envname> python=3.9```

- ### activate and initialize virtual env
    - ```conda activate <envname>```
- Download the model and store it in the same repository as the chatbot.py

- RUN ```pip install -r requirements.txt```:
  - This line installs the Python dependencies listed in the ```requirements.txt``` file using pip.
To run Streamlit app, run the following:
``` shell
streamlit run chatbot.py
```
