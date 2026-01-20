import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------- CONFIGURACIÓN ----------------
st.set_page_config(page_title="Mini ChatGPT", page_icon="🤖")
st.title("🤖 Mini ChatGPT con Videos")

# ---------------- ESTILOS ----------------
st.markdown("""
<style>
.chat-container {
    max-width: 700px;
    margin: auto;
}
.user-bubble {
    background-color: #DCF8C6;
    padding: 10px 14px;
    border-radius: 15px;
    margin: 8px 0;
    text-align: right;
}
.bot-bubble {
    background-color: #F1F0F0;
    padding: 10px 14px;
    border-radius: 15px;
    margin: 8px 0;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# ---------------- VIDEOS DE YOUTUBE ----------------
VIDEO_RESPUESTAS = {
    "que es python": {
        "texto": "Aquí tienes un video para aprender qué es Python 🐍",
        "video": "https://www.youtube.com/watch?v=rfscVS0vtbw"
    },
    "que es inteligencia artificial": {
        "texto": "Este video explica la Inteligencia Artificial 🤖",
        "video": "https://www.youtube.com/watch?v=2ePf9rue1Ao"
    },
    "que es machine learning": {
        "texto": "Aprende Machine Learning con este video 📊",
        "video": "https://www.youtube.com/watch?v=ukzFI9rgwfU"
    },
    "como crear un chatbot": {
        "texto": "Este video te enseña cómo crear un chatbot 💬",
        "video": "https://www.youtube.com/watch?v=JMUxmLyrhSk"
    }
}

# ---------------- CARGAR MODELO ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- ESTADOS ----------------
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- MOSTRAR CHAT ----------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-bubble'>🧑 {msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot-bubble'>🤖 {msg['content']}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.text_input("Escribe tu mensaje:")

# ---------------- BOTÓN ENVIAR ----------------
if st.button("Enviar") and user_input:
    user_text = user_input.lower().strip()

    # Guardar mensaje del usuario
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # ----- RESPUESTA CON VIDEO -----
    if user_text in VIDEO_RESPUESTAS:
        bot_text = VIDEO_RESPUESTAS[user_text]["texto"]

        st.session_state.messages.append({
            "role": "bot",
            "content": bot_text
        })

        st.video(VIDEO_RESPUESTAS[user_text]["video"])

    # ----- RESPUESTA NORMAL -----
    else:
        new_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token,
            return_tensors="pt"
        )

        if st.session_state.chat_history_ids is not None:
            bot_input_ids = torch.cat(
                [st.session_state.chat_history_ids, new_input_ids], dim=-1
            )
        else:
            bot_input_ids = new_input_ids

        st.session_state.chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(
            st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        st.session_state.messages.append({
            "role": "bot",
            "content": response
        })


            st.markdown(f"**🤖 Bot:** {response}")

