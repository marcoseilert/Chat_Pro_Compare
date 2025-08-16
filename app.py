import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
import requests
from pathlib import Path
import logging
import json
import re
import threading
import hashlib
import os

# --- Configuração Inicial ---

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração da página
st.set_page_config(
    page_title="TripleLLM Arena - Multi-Model Chat",
    page_icon="🖥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Gerenciamento de Sessão e Armazenamento ---

# Usar session_state para armazenamento temporário ao invés de arquivos
# Isso é mais seguro e apropriado para apps públicos

def get_session_id() -> str:
    """
    Gera um ID de sessão único baseado em session_state.
    Para apps públicos, não devemos depender de autenticação.
    """
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

# --- Gerenciamento de Modelos ---

def get_initial_models():
    """Retorna uma estrutura de modelos padrão."""
    initial_free = [
        {'id': 'qwen/qwen3-235b-a22b:free', 'name': 'Qwen3-235b-a22b', 'company': 'Qwen'},
    ]
    initial_paid = [
        {'id': 'google/gemini-2.5-flash', 'name': 'Gemini 2.5 Flash', 'company': 'Google'},
    ]
    return initial_free, initial_paid

def fetch_and_classify_models() -> tuple[list | None, list | None]:
    """Busca modelos da API OpenRouter."""
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {e}")
        logging.error(f"API request error: {e}")
        return None, None

    one_year_ago = datetime.now() - pd.Timedelta(days=365)
    free_models_data, paid_models_data = [], []

    for model in data.get('data', []):
        model_id = model.get('id')
        model_name = model.get('name')
        created_timestamp = model.get('created', 0)

        if not all([model_id, model_name, created_timestamp]):
            continue
      
        if datetime.fromtimestamp(created_timestamp) < one_year_ago:
            continue

        company_name = model_name.split(':')[0].strip() if ':' in model_name else "Others"
        model_info = {'id': model_id, 'name': model_name, 'company': company_name, 'created': created_timestamp}

        if model.get('pricing', {}).get('prompt') == '0':
            free_models_data.append(model_info)
        else:
            paid_models_data.append(model_info)

    free_models_data.sort(key=lambda item: item['created'], reverse=True)
    paid_models_data.sort(key=lambda item: item['created'], reverse=True)
  
    return free_models_data, paid_models_data

def ordenar_empresas(empresas: list[str]) -> list[str]:
    """Ordena empresas por relevância em IA."""
    principais_em_ia = ["Google", "OpenAI", "xAI", "Anthropic", "Qwen", "DeepSeek", "Perplexity", "Meta"]
    empresas = list(dict.fromkeys(empresas))
    top = [e for e in principais_em_ia if e in empresas]
    resto = sorted([e for e in empresas if e not in principais_em_ia])
    return top + resto

def clean_model_name(name: str) -> str:
    """Remove prefixo da empresa do nome do modelo."""
    if ':' in name:
        return name.split(':', 1)[1].strip()
    return name

# Inicialização de modelos
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_models():
    """Carrega modelos com cache para melhor performance."""
    free_models, paid_models = get_initial_models()
    
    # Tentar buscar modelos atualizados da API
    try:
        new_free, new_paid = fetch_and_classify_models()
        if new_free and new_paid:
            free_models = new_free  # Limitar para não sobrecarregar
            paid_models = new_paid
    except:
        pass  # Usar modelos padrão se API falhar
    
    for model_list in [free_models, paid_models]:
        for model in model_list:
            model['name'] = clean_model_name(model['name'])
    
    return free_models, paid_models

FREE_MODELS, PAID_MODELS = load_models()
ALL_MODELS = FREE_MODELS + PAID_MODELS
MODEL_NAME_MAP = {model['id']: model['name'] for model in ALL_MODELS}
MODEL_ID_MAP = {model['name']: model['id'] for model in ALL_MODELS}
ALL_COMPANIES = ordenar_empresas(list(set(model['company'] for model in ALL_MODELS)))

DEFAULT_MODEL_ID = "google/gemini-flash-1.5"
if not any(m['id'] == DEFAULT_MODEL_ID for m in ALL_MODELS):
    DEFAULT_MODEL_ID = ALL_MODELS[0]['id'] if ALL_MODELS else None

DEFAULT_MODEL_COMPANY = next((m['company'] for m in ALL_MODELS if m['id'] == DEFAULT_MODEL_ID), None)

def filter_models(show_free: bool, show_paid: bool, selected_companies: list) -> list:
    """Filtra modelos baseado nas preferências."""
    filtered_list = []
    if not selected_companies: return []
    if show_free: filtered_list.extend(m for m in FREE_MODELS if m['company'] in selected_companies)
    if show_paid: filtered_list.extend(m for m in PAID_MODELS if m['company'] in selected_companies)
    return filtered_list

# --- Estilos CSS ---
st.markdown("""
<style>
    .main { background-color: #f0f2f5; font-family: 'Inter', sans-serif; }
    .stMultiSelect [data-baseweb="tag"] { height: fit-content; }
    .stMultiSelect [data-baseweb="tag"] span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 150px; }
    h3 { margin-top: 1em; }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Funções de Sessão ---

def initialize_session_state():
    """Inicializa variáveis de sessão."""
    if 'conversation_id' not in st.session_state: 
        st.session_state.conversation_id = str(uuid.uuid4())
    if 'messages' not in st.session_state: 
        st.session_state.messages = []
    if 'selected_model_ids' not in st.session_state:
        st.session_state.selected_model_ids = [DEFAULT_MODEL_ID] if DEFAULT_MODEL_ID else []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'show_free_models' not in st.session_state: 
        st.session_state.show_free_models = True
    if 'show_paid_models' not in st.session_state: 
        st.session_state.show_paid_models = False  # Padrão para false em app público
    if 'selected_companies' not in st.session_state:
        st.session_state.selected_companies = [DEFAULT_MODEL_COMPANY] if DEFAULT_MODEL_COMPANY else []
    if 'web_search_enabled' not in st.session_state: 
        st.session_state.web_search_enabled = False
    if 'saved_conversations' not in st.session_state:
        st.session_state.saved_conversations = []
    if 'api_key_hash' not in st.session_state:
        st.session_state.api_key_hash = None

def validate_api_key(api_key: str) -> bool:
    """Valida formato básico da chave API."""
    if not api_key:
        return False
    # OpenRouter keys geralmente começam com 'sk-or-'
    return len(api_key) > 20

def hash_api_key(api_key: str) -> str:
    """Cria hash da API key para verificação sem armazenar a chave."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def call_openrouter_api(model_id: str, api_key: str, conversation_history: list, web_search_enabled: bool = False) -> str:
    """Chama a API OpenRouter."""
    if not api_key: 
        return "Error: API key not configured."
    if not model_id: 
        return "Error: No model selected."
    if not conversation_history: 
        return "Error: Empty conversation history."

    effective_model_id = f"{model_id}:online" if web_search_enabled else model_id
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://share.streamlit.io",
        "X-Title": "TripleLLM Arena"
    }
  
    api_messages = []
    for msg in conversation_history:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            api_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            if isinstance(content, dict):
                first_response = next(iter(content.values()), None)
                if first_response:
                    api_messages.append({"role": "assistant", "content": first_response})
            else:
                api_messages.append({"role": "assistant", "content": content})

    # Limitar tokens para controlar custos
    payload = {
        "model": effective_model_id, 
        "messages": api_messages, 
        "max_tokens": 8048,  # Reduzido para apps públicos
        "temperature": 0.7, 
        "top_p": 0.9
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)  # Timeout reduzido
        response.raise_for_status()
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            return str(content)
        return "Error: Unexpected API response format."
    except requests.exceptions.Timeout:
        return "Error: Request timeout. Please try again."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return "Error: Invalid API key. Please check your OpenRouter API key."
        elif e.response.status_code == 429:
            return "Error: Rate limit exceeded. Please wait a moment and try again."
        error_details = e.response.json().get("error", {}).get("message", str(e))
        return f"Error: {error_details}"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Funções de Conversa (Armazenamento em Memória) ---

def save_conversation_to_memory():
    """Salva conversa na memória da sessão."""
    if st.session_state.messages:
        conversation_data = {
            'id': st.session_state.conversation_id,
            'messages': st.session_state.messages.copy(),
            'model_ids': st.session_state.selected_model_ids.copy(),
            'timestamp': datetime.now(),
            'preview': st.session_state.messages[0]['content'][:50] + "..." if st.session_state.messages else "Empty"
        }
        
        # Limitar número de conversas salvas
        MAX_SAVED_CONVERSATIONS = 10
        st.session_state.saved_conversations = [
            conv for conv in st.session_state.saved_conversations 
            if conv['id'] != st.session_state.conversation_id
        ]
        st.session_state.saved_conversations.insert(0, conversation_data)
        st.session_state.saved_conversations = st.session_state.saved_conversations[:MAX_SAVED_CONVERSATIONS]

def load_conversation_from_memory(conversation_id: str):
    """Carrega conversa da memória."""
    for conv in st.session_state.saved_conversations:
        if conv['id'] == conversation_id:
            st.session_state.conversation_id = conv['id']
            st.session_state.messages = conv['messages'].copy()
            st.session_state.selected_model_ids = conv['model_ids'].copy()
            return True
    return False

def delete_conversation_from_memory(conversation_id: str):
    """Remove conversa da memória."""
    st.session_state.saved_conversations = [
        conv for conv in st.session_state.saved_conversations 
        if conv['id'] != conversation_id
    ]
    if st.session_state.conversation_id == conversation_id:
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())

# --- Inicialização ---
initialize_session_state()

# --- Barra Lateral ---
with st.sidebar:
    st.header("🖥 TripleLLM Arena")
    
    # Informações sobre o app
    with st.expander("ℹ️ About this App", expanded=False):
        st.markdown("""
        **TripleLLM Arena** allows you to:
        - Compare responses from multiple AI models simultaneously
        - Test different models with the same prompt
        - Save conversations in your browser session
        
        **How to use:**
        1. Get your [OpenRouter API key](https://openrouter.ai/keys)
        2. Refresh Models list
        3. Select up to 3 models to compare
        3. Type your message and see responses side by side
        
        **Note:** This is a community app. Your API key is only used for this session and is not stored.
        """)
    
    st.divider()

    # Nova conversa
    if st.button("🆕 New Conversation", use_container_width=True, type="primary"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.selected_model_ids = [DEFAULT_MODEL_ID] if DEFAULT_MODEL_ID else []
        st.toast("New conversation started!", icon="✨")
        st.rerun()

    st.divider()
    
    # Filtros de modelos
    st.subheader("🎯 Model Filters")
    col_f1, col_f2 = st.columns(2)
    st.session_state.show_free_models = col_f1.checkbox("Free", value=st.session_state.show_free_models)
    st.session_state.show_paid_models = col_f2.checkbox("Paid", value=st.session_state.show_paid_models)
    
    if st.session_state.show_paid_models:
        st.caption("⚠️ Paid models require credits in your OpenRouter account")

    temp_models = filter_models(True, True, ALL_COMPANIES)
    available_companies = ordenar_empresas(list(set(m['company'] for m in temp_models)))
    
    st.session_state.selected_companies = st.multiselect(
        "Companies", 
        options=available_companies, 
        default=st.session_state.selected_companies,
        help="Select AI companies to show their models"
    )

    st.divider()
    
    # Seleção de modelos
    st.subheader("🖥 Model Selection")
    #st.caption("Select up to 3 models to compare", help="Choose up to 3 models to compare their responses")
    
    available_models = filter_models(
        st.session_state.show_free_models,
        st.session_state.show_paid_models,
        st.session_state.selected_companies
    )
    available_model_names = [m['name'] for m in available_models]
    
    current_selection_names = [MODEL_NAME_MAP.get(mid) for mid in st.session_state.selected_model_ids if mid in MODEL_NAME_MAP]
    valid_current_selection = [name for name in current_selection_names if name in available_model_names]

    selected_friendly_names = st.multiselect(
        #"AI Models",
        options=available_model_names,
        default=valid_current_selection,
        max_selections=3,
        key="model_selector",
        help="Choose up to 3 models to compare their responses"
    )
    
    st.session_state.selected_model_ids = [MODEL_ID_MAP[name] for name in selected_friendly_names if name in MODEL_ID_MAP]

    if not st.session_state.selected_model_ids:
        st.warning("⚠️ Please select at least one model")

    st.divider()

    # Web search toggle
    st.subheader("🌐 Web Search")
    st.session_state.web_search_enabled = st.toggle(
        "Enable web search", 
        value=st.session_state.web_search_enabled,
        help="Allow models to search the web for current information (may increase response time)"
    )

    st.divider()

    # Botão para atualizar modelos
    with st.expander("🔧 Refresh Model List"):
        if st.button("🔄 Refresh", use_container_width=True):
            with st.spinner("Fetching latest models..."):
                st.cache_data.clear()
                st.toast("Model list refreshed!", icon="✅")
                st.rerun()

    st.divider()
    
    # Conversas salvas (em memória)
    with st.expander("💾 Saved Conversations", expanded=False):
        if not st.session_state.saved_conversations:
            st.caption("No saved conversations in this session")
        else:
            st.caption(f"Showing {len(st.session_state.saved_conversations)} recent conversations")
            for conv in st.session_state.saved_conversations:
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    if st.button(
                        f"📝 {conv['preview']}", 
                        key=f"load_{conv['id']}", 
                        use_container_width=True
                    ):
                        if load_conversation_from_memory(conv['id']):
                            st.toast("Conversation loaded!", icon="📂")
                            st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{conv['id']}", help="Delete"):
                        delete_conversation_from_memory(conv['id'])
                        st.rerun()

    st.divider()
    
    # API Key input
    st.subheader("🔑 API Configuration")
    
    api_key_input = st.text_input(
        "OpenRouter API Key", 
        type="password", 
        value=st.session_state.api_key,
        placeholder="sk-or-...",
        help="Your API key is only used for this session"
    )
    
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        if validate_api_key(api_key_input):
            st.session_state.api_key_hash = hash_api_key(api_key_input)
            st.success("✅ API key format valid")
        elif api_key_input:
            st.error("❌ Invalid API key format")
    
    if not st.session_state.api_key:
        st.info("👉 [Get your OpenRouter API key](https://openrouter.ai/keys)")
        st.warning("⚠️ You need an API key to use this app")
    

# --- Área Principal do Chat ---
st.title("🖥 TripleLLM Arena")
st.caption("Compare responses from multiple AI models side by side")

# Mostrar modelos selecionados
if st.session_state.selected_model_ids:
    model_names_str = ", ".join([MODEL_NAME_MAP.get(mid, "Unknown") for mid in st.session_state.selected_model_ids])
    st.info(f"**Selected Models:** {model_names_str}")
else:
    st.warning("👈 Please select at least one model from the sidebar to start")

# Mostrar mensagens
if not st.session_state.messages:
    # Welcome message
    st.markdown("""
    ### 👋 Welcome to TripleLLM Arena!
    
    This tool lets you compare responses from different AI models simultaneously.
    
    **Quick Start:**
    1. Add your [OpenRouter API key](https://openrouter.ai/keys) in the sidebar
    2. Refresh Models list 
    3. Select up to 3 models to compare responses
    3. Type your message below
    4. See how different models respond to the same prompt!
    
    **Tips:**
    - Free models are great for testing
    - Enable web search for current information
    - Your conversations are saved during this session
    """)

# Exibir histórico de mensagens
for message in st.session_state.messages:
    role = message.get('role')
    content = message.get('content')
    
    if role == "user":
        with st.chat_message(name="user", avatar="👤"):
            st.markdown(content)
    
    elif role == "assistant":
        if isinstance(content, dict):
            num_models = len(content)
            cols = st.columns(num_models)
            for i, (model_id, response_text) in enumerate(content.items()):
                with cols[i]:
                    with st.container():
                        st.markdown(f"**{MODEL_NAME_MAP.get(model_id, 'Model')}**")
                        with st.chat_message(name="assistant", avatar="🤖"):
                            st.markdown(response_text)
        else:
            with st.chat_message(name="assistant", avatar="🤖"):
                st.markdown(content)

# Input de chat
input_disabled = not st.session_state.api_key or not st.session_state.selected_model_ids

if input_disabled:
    st.info("💡 Add your API key and select models to start chatting")

prompt = st.chat_input(
    "Type your message here...",
    key="prompt_input",
    disabled=input_disabled,
)

# Processar nova mensagem
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Gerar respostas
if st.session_state.messages and st.session_state.messages[-1]['role'] == 'user':
    
    with st.spinner("🤔 Models are thinking..."):
        assistant_responses = {}
        threads = []
        lock = threading.Lock()

        def get_model_response(model_id, api_key, conversation_history, web_search_enabled):
            response = call_openrouter_api(
                model_id=model_id,
                api_key=api_key,
                conversation_history=conversation_history,
                web_search_enabled=web_search_enabled
            )
            with lock:
                assistant_responses[model_id] = response

        api_key = st.session_state.api_key
        conversation_history = st.session_state.messages
        web_search_enabled = st.session_state.web_search_enabled

        # Criar threads para chamadas paralelas
        for model_id in st.session_state.selected_model_ids:
            thread = threading.Thread(
                target=get_model_response, 
                args=(model_id, api_key, conversation_history, web_search_enabled)
            )
            threads.append(thread)
            thread.start()

        # Aguardar todas as respostas
        for thread in threads:
            thread.join()

    # Ordenar respostas
    ordered_responses = {
        mid: assistant_responses.get(mid, "Error: No response received.") 
        for mid in st.session_state.selected_model_ids
    }

    st.session_state.messages.append({"role": "assistant", "content": ordered_responses})
    
    # Salvar conversa na memória
    save_conversation_to_memory()
    
    # Desabilitar web search após uso
    if st.session_state.web_search_enabled:
        st.session_state.web_search_enabled = False

    st.rerun()

# Footer
st.divider()
st.caption("Built with AI 🤖 By [Marcos Eilert](https://www.linkedin.com/in/marcos-eilert-%F0%9F%92%B9%F0%9F%93%88%F0%9F%93%8A-61b66722/) using Streamlit | Powered by [OpenRouter](https://openrouter.ai)")
