# Importation des bibliothèques
import streamlit as st
import os
import base64
import pyttsx3
import time
import pandas as pd
import streamlit.components.v1 as components
import speech_recognition as sr
import whisper

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict
from langgraph.graph import StateGraph, END

# =========================================================
# 1. CONFIGURATION
# =========================================================
st.set_page_config(page_title="Chatbot ENSAJ", page_icon="🎓", layout="wide")

if "messages" not in st.session_state: st.session_state.messages = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "model" not in st.session_state: st.session_state.model = None

# =========================================================
# 2. FONCTIONS (VOIX, DATA, AVATAR)
# =========================================================

def speak_offline(text):
    """Synthèse vocale directe"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 175) # Un peu plus lent pour être naturel
        voices = engine.getProperty('voices')
        for v in voices:
            if "fr" in v.id.lower(): engine.setProperty('voice', v.id); break
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except: pass

@st.cache_resource
def load_whisper_model(): return whisper.load_model("base")

def get_avatar_base64():
    if os.path.exists("mon_avatar.glb"):
        with open("mon_avatar.glb", 'rb') as f: return base64.b64encode(f.read()).decode()
    return None

def show_avatar(mode="sidebar"):
    """Affiche l'avatar. Mode 'sidebar' (petit) ou 'main' (géant/zoomé)"""
    bin_str = get_avatar_base64()
    if not bin_str: return st.warning("Avatar introuvable.")

    if mode == "sidebar":
        # Vue corps entier (Petit)
        height = "280px"
        orbit = "0deg 90deg 2.5m"
        target = "0m 1m 0m" # Centre du corps
    else:
        # Vue Portrait (Grand & Zoomé sur le visage)
        height = "600px" # Très grand
        orbit = "0deg 85deg 1.3m" # Zoom avant (1.3m au lieu de 2.5m)
        target = "0m 1.6m 0m" # Focus sur la TÊTE (1.6m de haut)

    html_code = f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.1.1/model-viewer.min.js"></script>
    <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
        <model-viewer 
            src="data:application/octet-stream;base64,{bin_str}" 
            camera-controls auto-rotate shadow-intensity="1"
            camera-orbit="{orbit}" 
            camera-target="{target}"
            field-of-view="30deg"
            style="width: 100%; height: {height}; background-color: transparent; border-radius: 20px;">
        </model-viewer>
    </div>
    """
    components.html(html_code, height=int(height.replace("px",""))+10)

def init_chromadb():
    """Charge la mémoire des documents"""
    persist_directory = "./chroma_db"
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    
    docs = []
    if not os.path.exists("data"): os.makedirs("data"); return None
    for f in os.listdir("data"):
        try:
            path = os.path.join("data", f)
            if f.endswith(".txt"): docs.append(open(path, encoding="utf-8").read())
            elif f.endswith(".csv"): docs.append(pd.read_csv(path).to_string())
        except: pass
    
    if not docs: return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).create_documents(docs)
    return Chroma.from_documents(chunks, embedding_function, persist_directory=persist_directory)

def listen_continuously():
    """Écoute le micro"""
    r = sr.Recognizer()
    try: mic = sr.Microphone(device_index=1) # <--- Vérifiez votre Index Micro
    except: mic = sr.Microphone()
    r.dynamic_energy_threshold = False
    r.energy_threshold = 400
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            st.toast("👂 J'écoute...", icon="🎤")
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        st.toast("🧠 Réflexion...", icon="💭")
        with open("input.wav", "wb") as f: f.write(audio.get_wav_data())
        return load_whisper_model().transcribe("input.wav")["text"]
    except: return None

# =========================================================
# 3. LOGIQUE INTELLIGENTE (AGENT)
# =========================================================
class AgentState(TypedDict): question: str; context: str; answer: str

def router(s):
    res = st.session_state.model.invoke(f"Question: {s['question']}. Si c'est 'bonjour/merci/ça va' réponds CHAT. Sinon RAG.").content.upper()
    return "rag" if "RAG" in res else "chat"

def rag(s):
    res = st.session_state.vector_db.similarity_search(s["question"], k=3)
    ctx = "\n".join([d.page_content for d in res])
    ans = st.session_state.model.invoke(f"Contexte: {ctx}. Question: {s['question']}").content
    return {"answer": ans, "context": ctx}

def chat(s):
    return {"answer": st.session_state.model.invoke(f"Réponds courtoisement à: {s['question']}").content, "context": "chat"}

def run_agent(q):
    wf = StateGraph(AgentState)
    wf.add_node("rag", rag); wf.add_node("chat", chat)
    wf.set_conditional_entry_point(router, {"rag": "rag", "chat": "chat"})
    wf.add_edge("rag", END); wf.add_edge("chat", END)
    return wf.compile().invoke({"question": q, "context": "", "answer": ""})

# =========================================================
# 4. INTERFACE UTILISATEUR (IMMERSIVE)
# =========================================================

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🎛️ Contrôle")
    mode = st.radio("Mode :", ["💬 Chat Texte", "📞 Appel Visio (Géant)"])
    
    st.divider()
    # En mode Chat, l'avatar reste petit à gauche
    if mode == "💬 Chat Texte":
        st.caption("Votre Assistant")
        show_avatar(mode="sidebar")
        
    if st.button("🗑️ Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()

# --- CONTENU PRINCIPAL ---
if not st.session_state.vector_db:
    st.title("🎓 Assistant ENSAJ")
    st.info("Veuillez lancer le système à gauche.")
    if st.sidebar.button("🚀 Lancer le moteur"):
        st.session_state.model = ChatOllama(model="llama3.2", temperature=0)
        st.session_state.vector_db = init_chromadb()
        st.rerun()

else:
    prompt = None

    # --- MODE 1 : APPEL VISIO (GÉANT) ---
    if mode == "📞 Appel Visio (Géant)":
        # Centrage complet
        c1, c2, c3 = st.columns([1, 8, 1])
        with c2:
            # L'avatar est appelé en mode 'main' -> Il sera grand et zoomé
            show_avatar(mode="main")
            
        st.info("🟢 Micro ouvert. Parlez naturellement...", icon="🎙️")
        
        # Écoute
        user_text = listen_continuously()
        if user_text: prompt = user_text

    # --- MODE 2 : CHAT TEXTE ---
    else:
        st.title("🎓 Chatbot ENSAJ")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if txt := st.chat_input("Votre question..."): prompt = txt

    # --- TRAITEMENT COMMUN ---
    if prompt:
        # User
        st.session_state.messages.append({"role": "user", "content": prompt})
        if mode == "💬 Chat Texte":
             with st.chat_message("user"): st.markdown(prompt)

        # Assistant
        with st.chat_message("assistant"): # Juste pour le spinner
             with st.spinner("..."):
                res = run_agent(prompt)["answer"]
                
                # Affichage
                if mode == "💬 Chat Texte": st.markdown(res)
                else: st.toast(f"🤖 {res[:60]}...", icon="💬")
                
                # Voix
                if mode == "📞 Appel Visio (Géant)": speak_offline(res)

        st.session_state.messages.append({"role": "assistant", "content": res})
        
        # Relance immédiate en appel pour fluidité
        if mode == "📞 Appel Visio (Géant)": st.rerun()