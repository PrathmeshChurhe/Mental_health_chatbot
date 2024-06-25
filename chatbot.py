from llama_cpp import Llama
import streamlit as st
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import PromptHelper
from typing import Optional, List, Mapping, Any
import pandas as pd

st.set_page_config(page_title='Mental Heallth chatbot', page_icon=':robot_face:', layout='wide')

MODEL_NAME = 'mellogpt.Q3_K_S.gguf'
MODEL_PATH = 'model_path'
KNOWLEDGE_BASE_FILE = "mentalhealth.csv"

NUM_THREADS = 8
MAX_INPUT_SIZE = 2048
NUM_OUTPUT = 256
CHUNK_OVERLAP_RATIO = 0.10

try:
    prompt_helper = PromptHelper(MAX_INPUT_SIZE, NUM_OUTPUT, CHUNK_OVERLAP_RATIO)
except Exception as e:
    CHUNK_OVERLAP_RATIO = 0.2
    prompt_helper = PromptHelper(MAX_INPUT_SIZE, NUM_OUTPUT, CHUNK_OVERLAP_RATIO)

embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

class CustomLLM(LLM):
    model_name = MODEL_NAME

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        p = f"Human: {prompt} Assistant: "
        prompt_length = len(p)
        llm = Llama(model_path=MODEL_PATH, n_threads=NUM_THREADS)
        try:
            output = llm(p, max_tokens=512, stop=["Human:"], echo=True)['choices'][0]['text']
            response = output[prompt_length:]
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error("An error occurred while processing your request. Please try again.")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

# Cache functions using the new methods
@st.cache_resource
def load_model():
    return CustomLLM()

@st.cache_data
def load_knowledge_base():
    df = pd.read_csv(KNOWLEDGE_BASE_FILE)
    return dict(zip(df['Questions'].str.lower(), df['Answers']))

def clear_convo():
    st.session_state['messages'] = []

def init():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

# Main function
if __name__ == '__main__':
    init()
    knowledge_base = load_knowledge_base()
    llm = load_model()

    clear_button = st.sidebar.button("Clear Conversation")
    if clear_button:
        clear_convo()

    user_input = st.text_input("Enter your query:", key="user_input")
    if user_input:
        user_input = user_input.lower()
        answer = knowledge_base.get(user_input)
        if answer:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            llm._call(prompt=user_input)

    for message in st.session_state.messages:
        with st.container():
            st.markdown(f"**{message['role'].title()}**: {message['content']}")

