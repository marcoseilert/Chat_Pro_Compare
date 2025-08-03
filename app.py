import streamlit as st
import pandas as pd
import os
import uuid
from datetime import datetime
import requests
from pathlib import Path
import logging
import json
import re
import threading

# --- Configuração Inicial ---

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração da página
st.set_page_config(
    page_title="Chat Pro - Comparador de Modelos",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Gerenciamento de Usuário e Diretórios ---

BASE_CONVERSATIONS_DIR = Path("user_conversations")
BASE_CONVERSATIONS_DIR.mkdir(exist_ok=True)
MODELS_FILE = Path("models_config.json")

def get_user_id() -> str | None:
    """
    Retorna o email do usuário logado no Streamlit de forma segura.
    Verifica st.user (novo) e st.experimental_user (antigo) para compatibilidade.
    """
    try:
        # A forma moderna e preferida
        if hasattr(st, 'user') and st.user and hasattr(st.user, 'email'):
            return st.user.email
        # Fallback para versões mais antigas
        elif hasattr(st, 'experimental_user') and st.experimental_user and hasattr(st.experimental_user, 'email'):
            return st.experimental_user.email
        return None
    except Exception:
        return None


def get_user_specific_dir(user_id: str) -> Path:
    """Cria e retorna um diretório seguro para o ID do usuário."""
    if not user_id:
        raise ValueError("User ID não pode ser vazio.")
    
    # Sanitiza o email para criar um nome de diretório válido
    sanitized_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', user_id)
    user_dir = BASE_CONVERSATIONS_DIR / sanitized_id
    user_dir.mkdir(exist_ok=True)
    return user_dir

# --- Gerenciamento da Chave de API por Usuário ---

def save_api_key(user_id: str, api_key: str):
    """Salva a chave de API de forma segura no diretório do usuário."""
    if not user_id or not api_key:
        return
    try:
        user_dir = get_user_specific_dir(user_id)
        key_file = user_dir / ".api_key_store"
        with open(key_file, 'w') as f:
            json.dump({"api_key": api_key}, f)
        logging.info(f"Chave de API salva para o usuário {user_id}.")
    except Exception as e:
        logging.error(f"Falha ao salvar a chave de API para {user_id}: {e}")
        st.error("Não foi possível salvar a chave de API.")

def load_api_key(user_id: str) -> str | None:
    """Carrega a chave de API do arquivo do usuário, se existir."""
    if not user_id:
        return None
    try:
        user_dir = get_user_specific_dir(user_id)
        key_file = user_dir / ".api_key_store"
        if key_file.exists():
            with open(key_file, 'r') as f:
                data = json.load(f)
                logging.info(f"Chave de API carregada para o usuário {user_id}.")
                return data.get("api_key")
    except Exception as e:
        logging.error(f"Falha ao carregar a chave de API para {user_id}: {e}")
    return None


# --- Constantes e Gerenciamento de Modelos (sem alterações) ---
def get_initial_models():
    """Retorna uma estrutura de modelos padrão se o arquivo não existir."""
    initial_free = [
        {'id': 'google/gemini-flash-1.5', 'name': 'Google: Gemini Flash 1.5', 'company': 'Google'},
        {'id': 'meta-llama/llama-3-8b-instruct:free', 'name': 'Meta: Llama 3 8B Instruct (Free)', 'company': 'Meta'},
        {'id': 'microsoft/wizardlm-2-8x22b', 'name': 'Microsoft: WizardLM-2 8x22B', 'company': 'Microsoft'},
        {'id': 'openai/gpt-4o-mini', 'name': 'OpenAI: GPT-4o Mini', 'company': 'OpenAI'}
    ]
    initial_paid = [
        {'id': 'anthropic/claude-3.5-sonnet', 'name': 'Anthropic: Claude 3.5 Sonnet', 'company': 'Anthropic'},
        {'id': 'openai/gpt-4o', 'name': 'OpenAI: GPT-4o', 'company': 'OpenAI'},
        {'id': 'google/gemini-1.5-pro', 'name': 'Google: Gemini 1.5 Pro', 'company': 'Google'}
    ]
    return initial_free, initial_paid

def save_models_to_file(free_models: list, paid_models: list):
    """Salva as listas de dicionários de modelos em um arquivo JSON."""
    try:
        with open(MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'free_models': free_models, 'paid_models': paid_models}, f, indent=4)
        logging.info(f"Modelos salvos em {MODELS_FILE}")
    except Exception as e:
        logging.error(f"Falha ao salvar o arquivo de modelos: {e}")
        st.error(f"Não foi possível salvar as alterações dos modelos: {e}")

def load_models_from_file() -> tuple[list, list]:
    """Carrega as listas de dicionários de modelos do arquivo JSON."""
    if not MODELS_FILE.exists():
        logging.info(f"Arquivo {MODELS_FILE} não encontrado. Criando com modelos padrão.")
        initial_free, initial_paid = get_initial_models()
        save_models_to_file(initial_free, initial_paid)
        return initial_free, initial_paid
    
    try:
        with open(MODELS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            free_models = [m for m in data.get('free_models', []) if 'id' in m and 'name' in m and 'company' in m]
            paid_models = [m for m in data.get('paid_models', []) if 'id' in m and 'name' in m and 'company' in m]
            logging.info(f"Modelos carregados de {MODELS_FILE}")
            return free_models, paid_models
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Erro ao ler {MODELS_FILE}: {e}. Recriando com modelos padrão.")
        st.warning(f"Arquivo de configuração de modelos corrompido. Restaurando para o padrão.")
        initial_free, initial_paid = get_initial_models()
        save_models_to_file(initial_free, initial_paid)
        return initial_free, initial_paid

def fetch_and_classify_models() -> tuple[list | None, list | None]:
    """Busca modelos da API, filtra e classifica."""
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar modelos: {e}")
        logging.error(f"Erro na requisição à API OpenRouter: {e}")
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

        company_name = model_name.split(':')[0].strip() if ':' in model_name else "Outros"
        model_info = {'id': model_id, 'name': model_name, 'company': company_name, 'created': created_timestamp}

        if model.get('pricing', {}).get('prompt') == '0':
            free_models_data.append(model_info)
        else:
            paid_models_data.append(model_info)

    free_models_data.sort(key=lambda item: item['created'], reverse=True)
    paid_models_data.sort(key=lambda item: item['created'], reverse=True)
    
    logging.info(f"Modelos buscados: {len(free_models_data)} gratuitos, {len(paid_models_data)} pagos.")
    return free_models_data, paid_models_data

def ordenar_empresas(empresas: list[str]) -> list[str]:
    """Ordena a lista de empresas com base na relevância em IA."""
    principais_em_ia = ["Google", "OpenAI", "xAI", "Anthropic", "MoonshotAI", "DeepSeek", "Qwen", "Perplexity", "Meta", "Bytedance", "MiniMax", "Mistral"]
    empresas = list(dict.fromkeys(empresas))
    top = [e for e in principais_em_ia if e in empresas]
    resto = sorted([e for e in empresas if e not in principais_em_ia])
    return top + resto

def clean_model_name(name: str) -> str:
    """Remove o prefixo 'Empresa: ' do nome do modelo para uma exibição mais limpa."""
    if ':' in name:
        return name.split(':', 1)[1].strip()
    return name

# Carregamento e Mapeamento de Modelos
FREE_MODELS, PAID_MODELS = load_models_from_file()

for model_list in [FREE_MODELS, PAID_MODELS]:
    for model in model_list:
        model['name'] = clean_model_name(model['name'])

ALL_MODELS = FREE_MODELS + PAID_MODELS
MODEL_NAME_MAP = {model['id']: model['name'] for model in ALL_MODELS}
MODEL_ID_MAP = {model['name']: model['id'] for model in ALL_MODELS}
ALL_COMPANIES = ordenar_empresas(list(set(model['company'] for model in ALL_MODELS)))

DEFAULT_MODEL_ID = "google/gemini-flash-1.5"
if not any(m['id'] == DEFAULT_MODEL_ID for m in ALL_MODELS):
    DEFAULT_MODEL_ID = ALL_MODELS[0]['id'] if ALL_MODELS else None

DEFAULT_MODEL_COMPANY = next((m['company'] for m in ALL_MODELS if m['id'] == DEFAULT_MODEL_ID), None)

def filter_models(show_free: bool, show_paid: bool, selected_companies: list) -> list:
    """Filtra a lista de dicionários de modelos com base nas preferências."""
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
</style>
""", unsafe_allow_html=True)

# --- Funções de Sessão e API ---

def initialize_session_state(user_id: str | None):
    """Inicializa as variáveis necessárias no estado da sessão."""
    if 'conversation_id' not in st.session_state: st.session_state.conversation_id = str(uuid.uuid4())
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'selected_model_ids' not in st.session_state:
        st.session_state.selected_model_ids = [DEFAULT_MODEL_ID] if DEFAULT_MODEL_ID else []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = load_api_key(user_id) if user_id else ""
    if 'confirm_delete_id' not in st.session_state: st.session_state.confirm_delete_id = None
    if 'show_free_models' not in st.session_state: st.session_state.show_free_models = True
    if 'show_paid_models' not in st.session_state: st.session_state.show_paid_models = True
    if 'selected_companies' not in st.session_state:
        st.session_state.selected_companies = [DEFAULT_MODEL_COMPANY] if DEFAULT_MODEL_COMPANY else []
    if 'web_search_enabled' not in st.session_state: st.session_state.web_search_enabled = False

def call_openrouter_api(model_id: str, api_key: str, conversation_history: list, web_search_enabled: bool = False) -> str:
    """Chama a API OpenRouter Chat Completions de forma segura."""
    if not api_key: return "Erro: Chave API não configurada."
    if not model_id: return "Erro: Modelo não selecionado."
    if not conversation_history: return "Erro: Histórico de conversa vazio."

    effective_model_id = f"{model_id}:online" if web_search_enabled else model_id
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://share.streamlit.io",
        "X-Title": "Streamlit Chat Pro Comparador"
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

    payload = {"model": effective_model_id, "messages": api_messages, "max_tokens": 4096, "temperature": 0.5, "top_p": 0.9}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            logging.info(f"API call para {effective_model_id} bem-sucedida.")
            return str(content)
        logging.error(f"Resposta inesperada da API ({effective_model_id}): {result}")
        return "Erro: Resposta da API em formato inesperado."
    except requests.exceptions.Timeout:
        return "Erro: Timeout. A requisição demorou muito."
    except requests.exceptions.HTTPError as e:
        error_details = e.response.json().get("error", {}).get("message", e.response.text)
        return f"Erro HTTP {e.response.status_code}: {error_details}"
    except Exception as e:
        logging.exception(f"Erro inesperado na API ({effective_model_id}): {e}")
        return f"Erro inesperado: {e}"

# --- Funções de Conversa (Atualizadas para Multi-usuário) ---

def save_conversation(user_id: str | None, conversation_id: str, messages: list, model_ids: list):
    """Salva a conversa apenas se houver um user_id (usuário logado)."""
    if not user_id:
        logging.info("Usuário não logado. Conversa não será salva.")
        return
    if not messages: return

    try:
        user_dir = get_user_specific_dir(user_id)
        filename = user_dir / f"{conversation_id}.parquet"
        
        df = pd.DataFrame(messages)
        if 'content' in df.columns:
            df['content'] = df['content'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
        df['timestamp'] = datetime.now().isoformat()
        df['model_ids'] = json.dumps(model_ids)
        
        df.to_parquet(filename, index=False)
        logging.info(f"Conversa {conversation_id} salva para o usuário {user_id}.")
    except Exception as e:
        st.error(f"Erro ao salvar a conversa: {e}")

def load_conversation_messages(user_id: str | None, conversation_id: str) -> tuple[list, list | None]:
    """Carrega mensagens da conversa de um usuário específico."""
    if not user_id:
        return [], None

    try:
        user_dir = get_user_specific_dir(user_id)
        filename = user_dir / f"{conversation_id}.parquet"
        if not filename.exists(): return [], None

        df = pd.read_parquet(filename)
        if 'content' in df.columns:
            df['content'] = df['content'].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith('{') else x)
        messages = df[['role', 'content']].to_dict('records')
        model_ids_json = df['model_ids'].iloc[0] if 'model_ids' in df.columns and not df.empty else '[]'
        model_ids = json.loads(model_ids_json)
        
        logging.info(f"Conversa {conversation_id} carregada para o usuário {user_id}.")
        return messages, model_ids
    except Exception as e:
        st.error(f"Erro ao ler arquivo da conversa: {e}")
        return [], None

def delete_conversation(user_id: str | None, conversation_id: str):
    """Exclui o arquivo de uma conversa de um usuário específico."""
    if not user_id:
        return

    try:
        user_dir = get_user_specific_dir(user_id)
        filename = user_dir / f"{conversation_id}.parquet"
        if filename.exists():
            filename.unlink()
            st.toast(f"Conversa excluída.", icon="🗑️")
            if st.session_state.conversation_id == conversation_id:
                st.session_state.messages = []
                st.session_state.conversation_id = str(uuid.uuid4())
                st.session_state.selected_model_ids = [DEFAULT_MODEL_ID] if DEFAULT_MODEL_ID else []
        st.session_state.confirm_delete_id = None
    except Exception as e:
        st.error(f"Erro ao excluir conversa: {e}")

def load_conversations_metadata(user_id: str | None) -> list:
    """Carrega metadados das conversas salvas para um usuário específico."""
    if not user_id:
        return []

    conversations = []
    try:
        user_dir = get_user_specific_dir(user_id)
        for file in user_dir.glob('*.parquet'):
            conversation_id = file.stem
            try:
                df = pd.read_parquet(file, columns=['content', 'timestamp', 'role'])
                if not df.empty:
                    user_messages = df[df['role'] == 'user']
                    preview = user_messages['content'].iloc[0][:40] + "..." if not user_messages.empty else "Conversa sem prompt"
                    timestamp_dt = datetime.fromisoformat(df['timestamp'].iloc[-1])
                    conversations.append({"id": conversation_id, "preview": preview, "timestamp_dt": timestamp_dt})
            except Exception as e:
                logging.error(f"Erro ao carregar metadados de {file.name} para {user_id}: {e}")
    except Exception as e:
        logging.error(f"Erro ao acessar diretório de conversas para {user_id}: {e}")

    conversations.sort(key=lambda x: x['timestamp_dt'], reverse=True)
    return conversations

# --- Inicialização e Lógica Principal ---
USER_ID = get_user_id()
initialize_session_state(USER_ID)

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.header("⚖️ Comparador de Modelos")
    
    # --- NOVO: SEÇÃO DE DEBUG DE AUTENTICAÇÃO ---
    with st.expander("🔍 Status da Autenticação (Debug)"):
        st.write("Esta seção ajuda a diagnosticar problemas de login.")
        try:
            st.write("Objeto `st.user` recebido:")
            # Mostra o objeto de usuário que o Streamlit está fornecendo ao script
            st.json(st.user, expanded=False) 
        except Exception as e:
            st.error(f"Não foi possível acessar st.user: {e}")
        
        st.write("ID de Usuário detectado (email):")
        st.info(f"`{USER_ID}`" if USER_ID else "`None` (Não detectado)")

    st.divider()

    # Lógica de exibição de status baseada no USER_ID
    if USER_ID:
        st.success(f"Logado como: **{USER_ID}**")
    else:
        st.warning("Você está em modo anônimo.")
        st.caption("Faça login no Streamlit e atualize a página para salvar suas conversas.")


    if st.button("Nova Conversa", use_container_width=True):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.confirm_delete_id = None
        st.session_state.selected_model_ids = [DEFAULT_MODEL_ID] if DEFAULT_MODEL_ID else []
        st.toast("Nova conversa iniciada!", icon="✨")
        st.rerun()

    st.divider()
    
    st.subheader("Filtros de Modelos")
    col_f1, col_f2 = st.columns(2)
    st.session_state.show_free_models = col_f1.checkbox("Gratuitos", value=st.session_state.show_free_models)
    st.session_state.show_paid_models = col_f2.checkbox("Pagos", value=st.session_state.show_paid_models)

    temp_models = filter_models(True, True, ALL_COMPANIES)
    available_companies = ordenar_empresas(list(set(m['company'] for m in temp_models)))
    
    st.session_state.selected_companies = st.multiselect(
        "Empresas", options=available_companies, default=st.session_state.selected_companies
    )

    st.divider()
    
    st.subheader("Seleção de Modelos (até 3)")
    available_models = filter_models(
        st.session_state.show_free_models,
        st.session_state.show_paid_models,
        st.session_state.selected_companies
    )
    available_model_names = [m['name'] for m in available_models]
    
    current_selection_names = [MODEL_NAME_MAP.get(mid) for mid in st.session_state.selected_model_ids if mid in MODEL_NAME_MAP]
    valid_current_selection = [name for name in current_selection_names if name in available_model_names]

    selected_friendly_names = st.multiselect(
        "Modelos de IA",
        options=available_model_names,
        default=valid_current_selection,
        max_selections=3,
        key="model_selector",
        help="Escolha até 3 modelos para comparar."
    )
    
    st.session_state.selected_model_ids = [MODEL_ID_MAP[name] for name in selected_friendly_names if name in MODEL_ID_MAP]

    st.divider()

    st.subheader("🌐 Consulta Web")
    st.session_state.web_search_enabled = st.toggle(
        "Ativar busca na web", value=st.session_state.web_search_enabled,
        help="Se ativado, os modelos terão acesso à internet."
    )

    st.divider()

    with st.expander("📂 Conversas Salvas", expanded=True):
        if not USER_ID:
            st.caption("Faça login para ver e salvar conversas.")
        else:
            conversations = load_conversations_metadata(USER_ID)
            if not conversations:
                st.caption("Nenhuma conversa salva.")
            else:
                for conv in conversations:
                    conv_id = conv['id']
                    is_confirming_delete = (st.session_state.get('confirm_delete_id') == conv_id)
                    col1, col2 = st.columns([0.8, 0.2])

                    with col1:
                        if st.button(f"{conv['preview']}", key=f"load_{conv_id}", use_container_width=True, disabled=is_confirming_delete):
                            messages, loaded_model_ids = load_conversation_messages(USER_ID, conv_id)
                            if messages is not None:
                                st.session_state.conversation_id = conv_id
                                st.session_state.messages = messages
                                st.session_state.selected_model_ids = loaded_model_ids if loaded_model_ids else []
                                st.session_state.confirm_delete_id = None
                                st.toast(f"Conversa '{conv['preview']}' carregada.", icon="📂")
                                st.rerun()
                    
                    with col2:
                        if is_confirming_delete:
                            if st.button("✔️", key=f"confirm_delete_{conv_id}", help="Confirmar exclusão", use_container_width=True):
                                delete_conversation(USER_ID, conv_id)
                                st.rerun()
                            if st.button("❌", key=f"cancel_delete_{conv_id}", help="Cancelar exclusão", use_container_width=True):
                                st.session_state.confirm_delete_id = None
                                st.rerun()
                        else:
                            if st.button("🗑️", key=f"delete_{conv_id}", help="Excluir esta conversa", use_container_width=True):
                                st.session_state.confirm_delete_id = conv_id
                                st.rerun()
    
    st.divider()

    with st.expander("🔧 Manutenção de Modelos"):
        st.subheader("Gerenciar Listas de Modelos")
        st.caption("Atualize as listas para obter os modelos mais recentes da API.")

        if st.button("🔄 Atualizar Automaticamente da API", type="primary", use_container_width=True):
            with st.spinner("Buscando modelos mais recentes..."):
                new_free, new_paid = fetch_and_classify_models()
                if new_free is not None and new_paid is not None:
                    save_models_to_file(new_free, new_paid)
                    st.toast("Listas de modelos atualizadas com sucesso!", icon="✅")
                    st.rerun()
                else:
                    st.toast("Falha ao atualizar os modelos.", icon="❌")

    st.divider()
    
    st.session_state.api_key = st.text_input(
        "🔑 Chave API OpenRouter", 
        type="password", 
        value=st.session_state.api_key, 
        help="Insira sua chave API do OpenRouter."
    )

    if USER_ID:
        if st.button("Salvar Chave", use_container_width=True):
            save_api_key(USER_ID, st.session_state.api_key)
            st.success("Chave de API salva para este usuário!")
        st.caption("Sua chave será armazenada no servidor.")

    if not st.session_state.api_key:
        st.warning("⚠️ Insira sua chave API para usar o chat.")
    st.markdown("[Obter chave OpenRouter](https://openrouter.ai/keys)", unsafe_allow_html=True)

# --- Área Principal do Chat ---
st.title("⚖️ Chat Pro: Comparador de Modelos")

model_names_str = ", ".join([MODEL_NAME_MAP.get(mid, "N/A") for mid in st.session_state.selected_model_ids])
st.caption(f"Conversando com: **{model_names_str}**")

if not st.session_state.messages:
    st.info("👋 Selecione até 3 modelos na barra lateral e digite sua mensagem abaixo para começar a comparar!")

for message in st.session_state.messages:
    role = message.get('role')
    content = message.get('content')
    
    if role == "user":
        with st.chat_message(name="user", avatar="🧑"):
            st.markdown(content)
    
    elif role == "assistant":
        if isinstance(content, dict):
            num_models = len(content)
            cols = st.columns(num_models)
            for i, (model_id, response_text) in enumerate(content.items()):
                with cols[i]:
                    with st.chat_message(name=MODEL_NAME_MAP.get(model_id, "Bot"), avatar="🤖"):
                        st.subheader(f"Resposta de `{MODEL_NAME_MAP.get(model_id)}`")
                        st.markdown(response_text)
        else:
             with st.chat_message(name="assistant", avatar="🤖"):
                st.markdown(content)

input_disabled = not st.session_state.api_key or not st.session_state.selected_model_ids
prompt = st.chat_input(
    "Digite sua mensagem para todos os modelos...",
    key="prompt_input",
    disabled=input_disabled,
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]['role'] == 'user':
    
    with st.spinner("Aguardando respostas dos modelos... ⏳"):
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

        for model_id in st.session_state.selected_model_ids:
            thread = threading.Thread(
                target=get_model_response, 
                args=(model_id, api_key, conversation_history, web_search_enabled)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    ordered_responses = {
        mid: assistant_responses.get(mid, "Erro: Resposta não recebida.") 
        for mid in st.session_state.selected_model_ids
    }

    st.session_state.messages.append({"role": "assistant", "content": ordered_responses})
    
    save_conversation(
        USER_ID,
        st.session_state.conversation_id, 
        st.session_state.messages, 
        st.session_state.selected_model_ids
    )

    if st.session_state.web_search_enabled:
        st.session_state.web_search_enabled = False

    st.rerun()
