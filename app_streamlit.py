import json
import pickle

import requests
import streamlit as st

HF_TOKEN = "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"
HF_MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# Cargar los recursos

@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo entrenado por el alumno desde un .pkl.
    Guardamos (vectorizador, modelo) juntos para poder transformar texto y predecir.
    """
    with open("modelo_faq.pkl", "rb") as f:
        vectorizador, modelo = pickle.load(f)
    return vectorizador, modelo


@st.cache_resource
def cargar_respuestas():
    """
    Carga las respuestas oficiales por categoría desde un JSON.
    Así el chatbot NO inventa: responde con el contenido de tu FAQ.
    """
    with open("faq_respuestas.json", "r", encoding="utf-8") as f:
        return json.load(f)


# Chatbot modelo local

def responder_faq(pregunta: str) -> tuple[str, str]:
    """
    1) Convierte la pregunta en números (vectorizador)
    2) Predice la etiqueta (modelo)
    3) Devuelve la respuesta oficial asociada a esa etiqueta
    """
    vectorizador, modelo = cargar_modelo()
    respuestas = cargar_respuestas()

    # Transformamos el texto del usuario al mismo formato que se entrenó
    x_vec = vectorizador.transform([pregunta])

    # Predicción de categoría
    etiqueta = modelo.predict(x_vec)[0]

    # Buscar respuesta oficial
    respuesta = respuestas.get(
        etiqueta,
        "No tengo esa respuesta todavía. Prueba a reformular la pregunta."
    )

    return etiqueta, respuesta


# Generación de imágenes con API

def generar_imagen_api(prompt: str) -> bytes:
    """
    Generación de imágenes (API cloud) con Hugging Face.
    - Token pegado en el código (simplificado).
    - Devuelve bytes de la imagen para st.image()
    """
    if HF_TOKEN.startswith("PEGA_AQUI"):
        raise ValueError("Debes pegar tu HF_TOKEN en el código.")

    prompt_limpio = " ".join(prompt.split())  # quita saltos de línea y espacios dobles
    url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt_limpio}

    r = requests.post(url, headers=headers, json=payload, timeout=120)

    # Si falla, enseñamos el texto del servidor para entender el motivo
    if r.status_code != 200:
        raise RuntimeError(f"HF API Error {r.status_code}: {r.text}")

    return r.content



# Interfaz basada en Rivalt

st.set_page_config(page_title="RIVALT Assistant IA")
st.markdown("""
<style>
/* 1) Fuente Rivalt: Lexend */
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@100..900&display=swap');

html, body, [class*="css"]  {
  font-family: 'Lexend', sans-serif !important;
}

/* 2) Fondo oscuro Rivalt */
.stApp {
  background: #121212;
}

/* 3) Contenedores / “tarjetas” */
div[data-testid="stHorizontalBlock"], 
div[data-testid="stVerticalBlock"] {
  gap: 1rem;
}

section[data-testid="stSidebar"] {
  background: #1e1e1e;
  border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* 4) Texto */
h1, h2, h3, p, label, span, div {
  color: #f2f2f2;
}

/* 5) Inputs */
/* Inputs estilo Rivalt */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {

  background-color: #1e1e1e !important;
  color: #f2f2f2 !important;

  border: 1px solid #FF6D14 !important;
  border-radius: 10px !important;
}

/* Focus (cuando escribes) */
.stTextInput input:focus,
.stTextArea textarea:focus,
div[data-baseweb="input"]:focus-within,
div[data-baseweb="textarea"]:focus-within {

  border: 1px solid #FF6D14 !important;
  box-shadow: 0 0 0 1px #FF6D14 !important;
}

/* 6) Botones estilo Rivalt (naranja #FF6D14) */
.stButton > button {
  background: #FF6D14 !important;
  color: #121212 !important;
  border: 0 !important;
  border-radius: 12px !important;
  padding: 0.6rem 1rem !important;
  font-weight: 700 !important;
}

.stButton > button:hover {
  filter: brightness(1.08);
  transform: translateY(-1px);
}

/* 7) Tabs más “Rivalt” */
button[data-baseweb="tab"] {
  color: #f2f2f2 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
  border-bottom: 2px solid #FF6D14 !important;
}

/* 8) Mensajes (success/warning/error) un poco más integrados */
div[data-testid="stAlert"] {
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: #1e1e1e;
}
/* =========================
   Tabs underline Rivalt
========================= */

/* Línea activa (underline que se mueve) */
button[data-baseweb="tab"][aria-selected="true"]::after {
  background-color: #FF6D14 !important;
}

/* Borde inferior activo (fallback) */
button[data-baseweb="tab"][aria-selected="true"] {
  border-bottom: 2px solid #FF6D14 !important;
}

/* Hover tabs */
button[data-baseweb="tab"]:hover {
  color: #FF6D14 !important;
}

/* Línea contenedor tabs (a veces gris) */
div[data-testid="stTabs"] {
  border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}

</style>
""", unsafe_allow_html=True)

st.title("Asistente IA")
st.write("Resuelve tus dudas o genera tu propia imagen sobre deportes")

tab1, tab2 = st.tabs(["💬 Chatbot FAQ", "🖼️ Genera tu imagen de deportes"])


# Pestaña chatbot

with tab1:
    st.subheader("💬 Preguntas frecuentes")
    st.write("Escribe tu duda sobre RIVALT y el asistente te responde con la FAQ oficial.")

    pregunta = st.text_input("Tu pregunta:")

    if st.button("Responder"):
        if not pregunta.strip():
            st.warning("Escribe una pregunta primero.")
        else:
            try:
                etiqueta, respuesta = responder_faq(pregunta)

                st.caption(f"Categoría detectada por el modelo: `{etiqueta}`")
                st.success(respuesta)

            except FileNotFoundError:
                st.error("No se encontró 'modelo_faq.pkl'. Ejecuta antes entrenar_modelo_faq.py.")
            except Exception as e:
                st.error(f"Error inesperado: {e}")


# Pestaña generación de imágenes

with tab2:
    st.subheader("🖼️ Generar imagen")
    st.write("Escribe una descripción y la app pedirá una imagen a una API cloud gratuita.")

    prompt = st.text_area(
        "Descripción de la imagen:",
        placeholder="Ej: Persona jugando a fútbol con sus amigos"
    )

    if st.button("Generar imagen"):
        if not prompt.strip():
            st.warning("Escribe un prompt primero.")
        else:
            try:
                img_bytes = generar_imagen_api(prompt)
                st.image(img_bytes, caption="Imagen generada por API")
            except Exception as e:
                st.error(f"Error generando la imagen: {e}")
