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

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Mini ChatGPT", page_icon="🤖")
st.title("🤖 Mini ChatGPT")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

user_input = st.text_input("Escribe tu mensaje")

if st.button("Enviar") and user_input:
    user_text = user_input.lower().strip()

    # 🎥 Si la pregunta tiene video
    if user_text in VIDEO_RESPUESTAS:
        st.markdown(f"**🤖 Bot:** {VIDEO_RESPUESTAS[user_text]['texto']}")
        st.video(VIDEO_RESPUESTAS[user_text]["video"])

    else:
        # 🤖 Chatbot normal
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

        st.markdown(f"**🤖 Bot:** {response}")


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

    st.write("🤖", response)
