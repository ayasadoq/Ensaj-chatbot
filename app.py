import streamlit as st
import sys
import traceback
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
import os

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chatbot ENSAJ - RAG",
    page_icon="🤖",
    layout="wide"
)

# Initialisation de l'état de session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_db" not in st.session_state:
    st.session_state.faiss_db = None
if "model" not in st.session_state:
    st.session_state.model = None
if "initialization_error" not in st.session_state:
    st.session_state.initialization_error = None

st.title("🤖 Chatbot ENSAJ - Système RAG")
st.markdown("Posez vos questions sur l'ENSAJ et obtenez des réponses basées sur les documents disponibles.")

# Fonction pour charger les documents (DÉFINITION AVANT LA SIDEBAR)
def load_documents():
    """Charge les documents .txt et .csv"""
    docs = []
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.warning(f"⚠️ Dossier '{data_dir}' créé. Ajoutez vos fichiers .txt ou .csv")
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        try:
            # Charger les fichiers .txt
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        docs.append(content)
                        st.success(f"✅ {filename}")
                    else:
                        st.warning(f"⚠️ {filename} (vide)")
            
            # Charger les fichiers .csv
            elif filename.endswith(".csv"):
                df = pd.read_csv(file_path, encoding="utf-8")
                # Convertir le DataFrame en texte lisible
                csv_text = df.to_string()
                if len(csv_text.strip()) > 0:
                    # Ajouter le nom du fichier comme contexte
                    full_content = f"Fichier: {filename}\n\n{csv_text}"
                    docs.append(full_content)
                    st.success(f"✅ {filename} ({len(df)} lignes)")
                else:
                    st.warning(f"⚠️ {filename} (vide)")
        
        except Exception as e:
            st.error(f"❌ {filename}: {str(e)}")
    
    return docs

# Fonction RAG
def ask_question(query):
    try:
        if not st.session_state.faiss_db:
            return "❌ Le système RAG n'est pas prêt."
        
        results = st.session_state.faiss_db.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content for doc in results])
        
        messages = [
            SystemMessage(content="""Tu es un chatbot spécialisé sur l'ENSAJ.
Réponds uniquement à partir du contexte fourni.
Si l'information n'est pas disponible, dis-le clairement.
Sois concis et précis."""),
            HumanMessage(content=f"Contexte:\n{context}\n\nQuestion: {query}")
        ]
        
        response = st.session_state.model.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"

with st.sidebar:
    st.header("📊 Informations du système")
    
    # Initialisation du système (ne se fait qu'une fois)
    if st.session_state.faiss_db is None and st.session_state.initialization_error is None:
        with st.status("Initialisation du système...", expanded=True) as status:
            try:
                # 1. Charger les documents
                st.write("📂 Chargement des documents...")
                docs = load_documents()
                
                if not docs:        
                    raise ValueError("Aucun document chargé. Ajoutez des fichiers .txt ou .csv dans le dossier 'data'.")    
                
                st.write(f"📄 **Nombre de documents chargés:** {len(docs)}")
                
                # 2. Découper le texte
                st.write("✂️ Découpage du texte...")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=80
                )
                chunks = splitter.create_documents(docs)
                st.write(f"📄 **Nombre de chunks créés:** {len(chunks)}")
                
                # 3. Embeddings + FAISS
                st.write("🔤 Initialisation des embeddings...")
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                test_embedding = embeddings.embed_query("test")
                st.write(f"✅ **Dimension des embeddings:** {len(test_embedding)}")
                
                st.write("🗂️ Création de la base vectorielle FAISS...")
                st.session_state.faiss_db = FAISS.from_documents(chunks, embeddings)
                st.success("✅ Base FAISS créée!")
                
                # 4. Modèle LLM
                st.write("🧠 Initialisation du modèle Mistral...")
                st.session_state.model = ChatOllama(
                    model="mistral",
                    temperature=0.1
                )
                st.success("✅ Modèle Mistral initialisé!")
                
                status.update(label="✅ Système prêt!", state="complete", expanded=False)
                
            except Exception as e:
                error_msg = f"❌ Erreur d'initialisation:\n{str(e)}\n\n{traceback.format_exc()}"
                st.session_state.initialization_error = error_msg
                st.error(error_msg)
                status.update(label="❌ Erreur d'initialisation", state="error")
    
    # Afficher l'erreur d'initialisation si elle existe
    if st.session_state.initialization_error:
        st.error("Le système n'a pas pu s'initialiser. Vérifiez la console pour plus de détails.")
    
    # Afficher les statistiques
    st.divider()
    st.subheader("📈 Statistiques")
    if st.session_state.faiss_db:
        st.write("✅ Système RAG prêt")
    else:
        st.write("⏳ Système en initialisation...")
    st.write(f"💬 Messages: {len(st.session_state.messages)}")
    
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utilisateur
if st.session_state.faiss_db is not None:
    if prompt := st.chat_input("Posez votre question sur l'ENSAJ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche..."):
                response = ask_question(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("⏳ En attente de l'initialisation du système...")