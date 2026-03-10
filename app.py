import streamlit as st
import spacy
import numpy as np
import random

from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import Matcher

# -----------------------------------------

@st.cache_resource
def load_model():
    return spacy.load("pt_core_news_sm")

nlp = load_model()

# -----------------------------------------
# Intenção + Entidade

matcher = Matcher(nlp.vocab)

pattern_pix = [
    {"LOWER": {"IN": ["pix"]}}
]

pattern_cartao = [
    {"LOWER": {"IN": ["cartão", "cartao"]}},
    {"LOWER": {"IN": ["crédito", "credito", "débito", "debito"]}, "OP": "?"}
]

pattern_dinheiro = [
    {"LOWER": {"IN": ["dinheiro", "espécie", "especie"]}}
]

matcher.add("PIX", [pattern_pix])
matcher.add("CARTAO", [pattern_cartao])
matcher.add("DINHEIRO", [pattern_dinheiro])

# -----------------------------------------
# Base de Conhecimento

faq = {
    "Qual é o horário de atendimento?": [
        "Atendemos de segunda a sexta, das 9h às 18h.",
        "Nosso horário é de segunda a sexta, das 9h às 18h.",
        "Funcionamos de seg a sex, entre 9h e 18h."
    ],

    "Como faço uma compra?": [
        "Acesse nosso site, escolha o produto e clique em Comprar.",
        "Entre no site, selecione o item e finalize a compra."
    ],

    "Quais são as formas de pagamento?": [
        "Aceitamos cartão de crédito, débito e PIX.",
        "Você pode pagar com cartão de crédito, débito e PIX."
    ],

    "Como posso cancelar meu pedido?": [
        "Para cancelar seu pedido, acesse sua conta → Meus Pedidos → Cancelar."
    ]
}

# -----------------------------------------

def get_embeddings(textos):
    return np.array([nlp(t).vector for t in textos])

perguntas = list(faq.keys())
emb_perguntas = get_embeddings(perguntas)

# -----------------------------------------

def detectar_pagamento(texto):

    doc = nlp(texto)

    matches = matcher(doc)

    resultados = []

    for match_id, start, end in matches:

        label = nlp.vocab.strings[match_id]

        span = doc[start:end]

        resultados.append((label, span.text))

    return resultados

# -----------------------------------------

def chatbot(pergunta_usuario):

    entidades = detectar_pagamento(pergunta_usuario)

    if entidades:

        tipo = entidades[0][0]

        if tipo == "PIX":
            return "Sim, aceitamos pagamento via PIX."

        if tipo == "CARTAO":
            return "Aceitamos cartão de crédito e débito."

        if tipo == "DINHEIRO":
            return "Sim, aceitamos pagamento em dinheiro."

    emb_usuario = nlp(pergunta_usuario).vector.reshape(1, -1)

    similaridades = cosine_similarity(emb_usuario, emb_perguntas)[0]

    melhor_idx = np.argmax(similaridades)

    if similaridades[melhor_idx] > 0.7:
        return random.choice(faq[perguntas[melhor_idx]])

    return "Desculpe, não entendi. Pode reformular?"


st.title("🤖 Chatbot Atendimento")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if pergunta := st.chat_input("Digite sua pergunta..."):

    st.session_state.messages.append({
        "role": "user",
        "content": pergunta
    })

    with st.chat_message("user"):
        st.write(pergunta)

    resposta = chatbot(pergunta)

    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta
    })

    with st.chat_message("assistant"):
        st.write(resposta)
