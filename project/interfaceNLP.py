import streamlit as st
import requests
import json
import re
import time
from datetime import datetime


# Configuración de la página
st.set_page_config(
    page_title="IAMAM LangGraph Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS (sin cambios)
st.markdown(
    """
    <style>
    /* [Estilos originales sin cambios] */
    </style>
    """, unsafe_allow_html=True
)

# Inicialización de variables de estado
if "session_id" not in st.session_state:
    session = st.query_params.get("session", ["default"])[0]
    st.session_state.session_id = str(hash(session))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_status" not in st.session_state:
    st.session_state.agent_status = "idle"  # idle, thinking, active

if "action_history" not in st.session_state:
    st.session_state.action_history = []

if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""

if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = {}

if "workflow_id" not in st.session_state:
    st.session_state.workflow_id = None

if "mode" not in st.session_state:
    st.session_state.mode = "Modo Chatbot"  # Modo por defecto

# Funciones para interactuar con el API del agente LangGraph
def send_chat_request(user_input, stream=False):
    """Envía la solicitud al API del agente LangGraph"""
    api_url = "http://localhost:8002/chat/completions"
    headers = {
        "x-api-key": "8ae2476c1ff1269327d677310a541c6cc9b9eab036d62e23c0fe2a4cbd216f81"
    }
    payload = {
        "messages": [{"role": "user", "content": user_input}],
        "temperature": st.session_state.get("temperature", 0.7),
        "max_tokens": 1024,
        "session_id": st.session_state.session_id
    }
    
    try:
        st.session_state.agent_status = "thinking"
        response = requests.post(api_url, json=payload, headers=headers, stream=stream, timeout=60)
        return response
    except requests.RequestException as e:
        st.session_state.agent_status = "idle"
        return {"error": str(e)}

def start_workflow(url):
    """Inicia un flujo de trabajo con la URL proporcionada"""
    api_url = "http://localhost:8002/workflows/new"
    headers = {
        "x-api-key": "8ae2476c1ff1269327d677310a541c6cc9b9eab036d62e23c0fe2a4cbd216f81"
    }
    payload = {
        "url": url,
        "execute": True
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            result = response.json()
            st.session_state.workflow_id = result["workflow_id"]
            st.session_state.messages.append({"role": "assistant", "content": f"Flujo de trabajo iniciado con ID: {result['workflow_id']}"})
        else:
            st.session_state.messages.append({"role": "error", "content": f"Error al iniciar el flujo: {response.status_code} - {response.text}"})
    except requests.RequestException as e:
        st.session_state.messages.append({"role": "error", "content": f"Error de conexión: {str(e)}"})
    finally:
        st.session_state.agent_status = "idle"

def execute_tool_action(tool_name, tool_params):
    """Ejecuta una acción específica en el agente LangGraph"""
    api_url = "http://localhost:8002/tools/execute"
    headers = {
        "x-api-key": "8ae2476c1ff1269327d677310a541c6cc9b9eab036d62e23c0fe2a4cbd216f81"
    }
    payload = {
        "session_id": st.session_state.session_id,
        "tool_name": tool_name,
        "parameters": tool_params
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error al ejecutar la herramienta: {response.status_code} - {response.text}"}
    except requests.RequestException as e:
        return {"error": f"Error de conexión: {str(e)}"}

def get_agent_state():
    """Obtiene el estado actual del agente LangGraph"""
    api_url = f"http://localhost:8002/sessions/{st.session_state.session_id}/state"
    headers = {
        "x-api-key": "8ae2476c1ff1269327d677310a541c6cc9b9eab036d62e23c0fe2a4cbd216f81"
    }
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error al obtener el estado: {response.status_code}"}
    except requests.RequestException as e:
        return {"error": f"Error de conexión: {str(e)}"}

# Nueva función para interactuar con la API de similitud semántica
def send_similarity_request(sentence1, sentence2):
    """Envía solicitud a la API de similitud semántica"""
    api_url = "http://app:8005/predict"
    #api_url = "http://localhost:8005/predict"  # URL de la API de similitud
    # headers = {
    #     "x-api-key": "8ae2476c1ff1269327d677310a541c6cc9b9eab036d62e23c0fe2a4cbd216f81"  # Misma clave que LangGraph
    # }
    payload = {
        "sentence1": sentence1,
        "sentence2": sentence2
    }
    
    try:
        st.session_state.agent_status = "thinking"
        response = requests.post(api_url, json=payload,  timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error en la API de similitud: {response.status_code} - {response.text}"}
    except requests.RequestException as e:
        return {"error": f"Error de conexión con la API de similitud: {str(e)}"}
    finally:
        st.session_state.agent_status = "idle"

# Componentes de la interfaz
def render_header():
    """Renderiza el encabezado de la aplicación"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 IAMAM LangGraph Agent")
    with col2:
        if st.session_state.agent_status == "idle":
            st.markdown('<div class="agent-status status-active">● Listo</div>', unsafe_allow_html=True)
        elif st.session_state.agent_status == "thinking":
            st.markdown('<div class="agent-status status-thinking">● Procesando</div>', unsafe_allow_html=True)

def render_sidebar():
    """Renderiza la barra lateral con opciones y controles adicionales"""
    with st.sidebar:
        st.header("Configuración")
        
        # Añadimos el nuevo modo de similitud semántica
        st.session_state.mode = st.selectbox(
            "Modo de operación",
            ["Modo Similitud Semántica"],
            index=["Modo Similitud Semántica"].index(st.session_state.mode)
        )
        
        agent_mode = st.selectbox(
            "Modo del agente",
            ["Normal", "Detallado", "Experto", "Creativo"],
            index=0
        )
        
        with st.expander("Opciones avanzadas"):
            temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            st.session_state["temperature"] = temperature
            st.checkbox("Mostrar trazas del agente", value=False, key="show_traces")
            st.checkbox("Modo verboso", value=False, key="verbose_mode")
        
        st.subheader("Herramientas disponibles")
        tool_expander = st.expander("Ver herramientas")
        with tool_expander:
            tools = [
                {"name": "Buscador web", "description": "Busca información en la web"},
                {"name": "Calculadora", "description": "Realiza cálculos matemáticos"},
                {"name": "Base de datos", "description": "Consulta la base de datos"},
                {"name": "Análisis de datos", "description": "Analiza conjuntos de datos"},
                {"name": "Similitud Semántica", "description": "Compara la similitud entre dos oraciones"}
            ]
            for tool in tools:
                st.markdown(f"**{tool['name']}**: {tool['description']}")
        
        st.subheader("Historial de acciones")
        with st.container():
            st.markdown('<div class="action-history">', unsafe_allow_html=True)
            if not st.session_state.action_history:
                st.markdown("<em>No hay acciones registradas</em>", unsafe_allow_html=True)
            else:
                for action in st.session_state.action_history[-5:]:
                    st.markdown(f'<div class="action-item">{action}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Acciones")
        if st.button("Reiniciar sesión", key="reset_session"):
            st.session_state.messages = []
            st.session_state.action_history = []
            st.session_state.agent_trace = {}
            st.session_state.workflow_id = None
            st.session_state.mode = "Modo Similitud Semántica"
            st.rerun()
        
        if st.download_button(
            label="Exportar conversación",
            data=json.dumps({
                "session_id": st.session_state.session_id,
                "messages": st.session_state.messages,
                "timestamp": datetime.now().isoformat()
            }, indent=2),
            file_name=f"conversation_{st.session_state.session_id}_{int(time.time())}.json",
            mime="application/json"
        ):
            st.success("Conversación exportada correctamente")

def render_messages():
    """Renderiza los mensajes de la conversación"""
    with st.container():
        st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                elif msg.get("role") == "assistant":
                    st.markdown(f'<div class="assistant-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                elif msg.get("role") == "tool":
                    st.markdown(f'<div class="tool-message"><span class="message-badge badge-tool">Herramienta</span> <strong>{msg["tool_name"]}</strong>: {msg["content"]}</div>', unsafe_allow_html=True)
                elif msg.get("role") == "thinking":
                    st.markdown(f'<div class="assistant-message"><span class="message-badge badge-thinking">Pensando</span> {msg["content"]}</div>', unsafe_allow_html=True)
                elif msg.get("role") == "error":
                    st.markdown(f'<div class="error-message"><span class="message-badge badge-error">Error</span> {msg["content"]}</div>', unsafe_allow_html=True)
                elif msg.get("role") == "similarity":
                    st.markdown(f'<div class="assistant-message"><span class="message-badge badge-tool">Similitud</span> {msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
def render_input_area():
    """Renderiza el área de entrada y botones"""
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    with st.container():
        with st.form(key="chat_form", clear_on_submit=True):
            if st.session_state.mode == "Modo Similitud Semántica":
                # Formulario para dos oraciones
                sentence1 = st.text_area("Oración 1:", placeholder="Ingresa la primera oración", key="sentence1")
                sentence2 = st.text_area("Oración 2:", placeholder="Ingresa la segunda oración", key="sentence2")
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write("")  # Espacio vacío para alineación
                with col2:
                    submit = st.form_submit_button("Comparar")
                with col3:
                    clear = st.form_submit_button("Limpiar")
                input_data = {"sentence1": sentence1, "sentence2": sentence2}  # Agrupar en un diccionario
            else:
                # Formulario original para Modo Chatbot y Modo Tareas
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    user_input = st.text_input(
                        "Escribe tu mensaje:", 
                        key="input", 
                        placeholder="Pregúntame algo o pídeme realizar una tarea...",
                        label_visibility="collapsed"
                    )
                with col2:
                    submit = st.form_submit_button("Enviar")
                with col3:
                    clear = st.form_submit_button("Limpiar")
                col_opts1, col_opts2 = st.columns(2)
                with col_opts1:
                    st.checkbox("Adjuntar contexto", value=False, key="attach_context")
                with col_opts2:
                    st.checkbox("Solo respuesta", value=False, key="response_only")
                input_data = user_input  # Solo un valor para otros modos
    st.markdown('</div>', unsafe_allow_html=True)
    
    return input_data, submit, clear  # Siempre devuelve 3 valores

def handle_user_input(input_data):
    """Procesa la entrada del usuario según el modo"""
    if st.session_state.mode == "Modo Similitud Semántica":
        sentence1 = input_data["sentence1"]
        sentence2 = input_data["sentence2"]
        print(input_data)
        if not sentence1 or not sentence2:
            st.session_state.messages.append({"role": "error", "content": "Por favor, ingresa ambas oraciones."})
            return
        
        st.session_state.messages.append({"role": "user", "content": f"Oración 1: {sentence1}\n Oración 2: {sentence2}"})
        response = send_similarity_request(sentence1, sentence2)
        
        if "error" in response:
            st.session_state.messages.append({"role": "error", "content": response["error"]})
        else:
            similarity_score = response.get("similarity", 0.0)
            st.session_state.messages.append({
                "role": "similarity",
                "content": f"Similitud semántica entre las oraciones: {similarity_score:.2f} (en una escala de 0 a 5)"
            })
        st.session_state.agent_status = "idle"
    else:
        user_input = input_data
        if user_input.lower() == "salir":
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": "¡Hasta luego! Espero verte pronto."})
            st.session_state.workflow_id = None
            return
        
        st.session_state.last_user_query = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Detectar comandos de flujo de trabajo en Modo Tareas
        if st.session_state.mode == "Modo Tareas":
            workflow_match = re.search(
                r"(?:ejecuta las pruebas a la siguiente url|validemos esta url|execute|prueba esta url)\s+(.+)",
                user_input,
                re.IGNORECASE
            )
            if workflow_match:
                url = workflow_match.group(1).strip()
                start_workflow(url)
                return
        
        # Procesar como conversación para Modo Chatbot o Modo Tareas
        response = send_chat_request(user_input, stream=False)
        
        if isinstance(response, dict) and "error" in response:
            st.session_state.messages.append({"role": "error", "content": f"Error al conectar con el servidor: {response['error']}"})
            st.session_state.agent_status = "idle"
            return
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                full_response = response_data["choices"][0]["message"]["content"]
                
                thinking_index = len(st.session_state.messages)
                st.session_state.messages.append({"role": "thinking", "content": "Procesando tu consulta..."})
                render_messages()
                
                if len(st.session_state.messages) > thinking_index:
                    st.session_state.messages.pop(thinking_index)
                
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.session_state.messages.append({"role": "error", "content": "No se recibió una respuesta válida del servidor"})
            except (json.JSONDecodeError, KeyError) as e:
                st.session_state.messages.append({"role": "error", "content": f"Error al procesar la respuesta: {str(e)}"})
        else:
            st.session_state.messages.append({"role": "error", "content": f"Error: {response.status_code} - {response.text}"})
        
        st.session_state.agent_status = "idle"

def render_visualizations():
    """Renderiza visualizaciones adicionales basadas en el contexto actual"""
    if st.session_state.get("show_traces", False) and st.session_state.agent_trace:
        with st.expander("Trazas del agente", expanded=False):
            st.json(st.session_state.agent_trace)

# Flujo principal de la aplicación
def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    header_container = st.container()
    with header_container:
        render_header()
    with st.sidebar:
        render_sidebar()
    chat_container = st.container()
    with chat_container:
        render_messages()
    vis_container = st.container()
    with vis_container:
        render_visualizations()
    input_container = st.container()
    input_data, submit, clear = render_input_area()
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit:
        handle_user_input(input_data)
        st.rerun()
    elif clear:
        st.session_state.messages = []
        st.session_state.workflow_id = None
        st.session_state.mode = "Modo Similitud Semántica"
        st.rerun()
    
    st.markdown("""
    <script>
        const chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        window.addEventListener('DOMContentLoaded', (event) => {
            const resizeObserver = new ResizeObserver(entries => {
                const chatContainer = document.getElementById('chat-container');
                if (chatContainer) {
                    chatContainer.style.maxHeight = `calc(100vh - 230px)`;
                }
            });
            resizeObserver.observe(document.body);
        });
    </script>
    """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()