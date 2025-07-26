import streamlit as st
import pandas as pd
import os
import uuid
from datetime import datetime
import requests
from pathlib import Path
import logging
import json
import threading
import time
import litellm

# --- Configuração Inicial ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="Chat Pro com Comparador de Modelos", page_icon="💬", layout="wide")

# --- Diretórios e Arquivos ---
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)
MODELS_FILE = Path("models_config.json")

# --- Carregamento e salva de modelos ---
def get_initial_models():
    initial_free = [
        {'id': 'deepseek/deepseek-chat-v3-0324:free', 'name': 'Deepseek Chat V3 0324 (Free)', 'company': 'Deepseek'},
        {'id': 'google/gemini-2.0-flash-exp:free', 'name': 'Gemini 2.0 Flash Exp (Free)', 'company': 'Google'}
    ]
    initial_paid = [
        {'id': 'anthropic/claude-3.7-sonnet', 'name': 'Claude 3.7 Sonnet', 'company': 'Anthropic'},
        {'id': 'openai/gpt-4o', 'name': 'GPT-4o', 'company': 'OpenAI'}
    ]
    return initial_free, initial_paid

def save_models_to_file(free_models, paid_models):
    with open(MODELS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'free_models': free_models, 'paid_models': paid_models}, f, indent=4)

def load_models_from_file():
    if not MODELS_FILE.exists():
        initial_free, initial_paid = get_initial_models()
        save_models_to_file(initial_free, initial_paid)
        return initial_free, initial_paid

    with open(MODELS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('free_models', []), data.get('paid_models', [])

FREE_MODELS, PAID_MODELS = load_models_from_file()
ALL_MODELS = FREE_MODELS + PAID_MODELS
MODEL_NAME_MAP = {model['id']: model['name'] for model in ALL_MODELS}
MODEL_ID_MAP = {model['name']: model['id'] for model in ALL_MODELS}

# --- Sessão ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# --- Função para chamada paralela de modelos ---
def get_model_response(model_id, prompt, results):
    try:
        start_time = time.time()
        response = litellm.completion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            api_key=st.session_state.api_key
        )
        end_time = time.time()
        content = response.choices[0].message.content
        results.append({"model": model_id, "content": content, "time": end_time - start_time})
    except Exception as e:
        results.append({"model": model_id, "content": f"Erro: {e}", "time": 0})

# --- Interface ---
st.title("🚀 Comparador de Modelos via OpenRouter")
st.caption("Compare as respostas dos principais modelos de linguagem.")

st.subheader("1. Informe sua chave da API OpenRouter")
st.session_state.api_key = st.text_input("API Key", type="password", value=st.session_state.api_key, help="Obtenha sua chave em https://openrouter.ai/keys")

if not st.session_state.api_key:
    st.warning("⚠️ Insira sua chave API para usar o comparador.")
    st.stop()

st.subheader("2. Escolha até 3 modelos para comparar")
selected_models = st.multiselect(
    "Modelos Disponíveis",
    options=[model['name'] for model in ALL_MODELS],
    default=[model['name'] for model in ALL_MODELS[:2]],
    max_selections=3
)

st.subheader("3. Digite seu prompt")
prompt = st.text_area("Prompt", "Explique a diferença entre aprendizado supervisionado e não supervisionado.", height=150)

if st.button("Comparar Respostas", type="primary"):
    if not selected_models:
        st.warning("Selecione ao menos um modelo.")
    elif not prompt.strip():
        st.warning("Digite um prompt válido.")
    else:
        st.info("Aguardando respostas dos modelos...")
        results = []
        threads = []
        for name in selected_models:
            model_id = MODEL_ID_MAP[name]
            thread = threading.Thread(target=get_model_response, args=(model_id, prompt, results))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        st.subheader("4. Resultados da Comparacão")
        cols = st.columns(len(results))
        for i, result in enumerate(results):
            with cols[i]:
                st.markdown(f"### {MODEL_NAME_MAP.get(result['model'], result['model'])}")
                st.markdown(f"*Tempo: {result['time']:.2f}s*")
                st.markdown("---")
                st.markdown(result['content'])
