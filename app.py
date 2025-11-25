import streamlit as st
import sys
import traceback
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
import os
import re

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chatbot ENSAJ - RAG Amélioré",
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
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

st.title("🤖 Chatbot ENSAJ - Système RAG Amélioré")
st.markdown("Posez vos questions sur l'ENSAJ et obtenez des réponses basées sur les documents disponibles.")

# Fonction pour nettoyer et normaliser le texte
def preprocess_text(text):
    """Nettoie et normalise le texte pour améliorer la recherche"""
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    # Normaliser la ponctuation
    text = re.sub(r'[\.\,;:\!?]+', ' ', text)
    return text.strip()

# Fonction pour charger les documents
def load_documents():
    """Charge les documents .txt et .csv avec prétraitement"""
    docs = []
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.warning(f"⚠️ Dossier '{data_dir}' créé. Ajoutez vos fichiers .txt ou .csv")
        return docs
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        try:
            # Charger les fichiers .txt
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        # Prétraiter le texte
                        content = preprocess_text(content)
                        # Ajouter des métadonnées pour améliorer la recherche
                        enhanced_content = f"Document: {filename}\n\n{content}"
                        docs.append(enhanced_content)
                        st.success(f"✅ {filename} ({len(content)} caractères)")
                    else:
                        st.warning(f"⚠️ {filename} (vide)")
            
            # Charger les fichiers .csv
            elif filename.endswith(".csv"):
                df = pd.read_csv(file_path, encoding="utf-8")
                # Convertir le DataFrame en texte lisible avec plus de contexte
                csv_text = ""
                for col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) > 0:
                        csv_text += f"Colonne {col}: {', '.join(map(str, unique_vals[:10]))}\n"
                
                if len(csv_text.strip()) > 0:
                    enhanced_content = f"Fichier CSV: {filename}\nColonnes: {', '.join(df.columns)}\n\nDonnées:\n{csv_text}"
                    docs.append(enhanced_content)
                    st.success(f"✅ {filename} ({len(df)} lignes, {len(df.columns)} colonnes)")
                else:
                    st.warning(f"⚠️ {filename} (vide)")
        
        except Exception as e:
            st.error(f"❌ {filename}: {str(e)}")
    
    return docs

# Fonction RAG améliorée
def ask_question(query):
    try:
        if not st.session_state.faiss_db:
            return "❌ Le système RAG n'est pas prêt."
        
        # Recherche étendue avec plus de chunks
        results = st.session_state.faiss_db.similarity_search(query, k=8)  # Augmenté de 4 à 8
        
        # Debug: afficher les chunks récupérés
        if st.session_state.debug_mode:
            st.sidebar.subheader("🔍 Debug - Chunks récupérés")
            for i, doc in enumerate(results):
                st.sidebar.write(f"**Chunk {i+1}:** {doc.page_content[:150]}...")
        
        context = "\n\n".join([doc.page_content for doc in results])
        
        # Prompt amélioré pour mieux utiliser le contexte
        system_prompt = """
Tu es un assistant expert spécialisé sur l'ENSAJ (École Nationale des Sciences Appliquées d'El Jadida).

INSTRUCTIONS CRITIQUES:
1. Analyse TRÈS ATTENTIVEMENT le contexte fourni
2. Si l'information exacte n'est pas trouvée, cherche des informations PARTIELLES ou APPROCHÉES
3. Pour les nombres et statistiques, sois particulièrement attentif aux chiffres dans le contexte
4. Si tu trouves des informations similaires mais pas exactes, fais une DÉDUCTION LOGIQUE
5. Ne dis JAMAIS "je ne sais pas" sans avoir minutieusement analysé chaque partie du contexte

EXEMPLES:
- Si on demande "nombre d'élèves" et que le contexte dit "1000 élèves ingénieurs", réponds "1000 élèves ingénieurs"
- Si on demande "effectif" et que le contexte dit "environ 1000 étudiants", réponds "environ 1000 étudiants"
- Si l'information est partielle, mentionne-le: "D'après les documents, [...]"

Réponds en français, sois précis et utile.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"CONTEXTE À ANALYSER:\n{context}\n\nQUESTION: {query}\n\nRéponds en t'appuyant STRICTEMENT sur le contexte fourni.")
        ]
        
        response = st.session_state.model.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"

# Fonction pour tester le système
def test_system():
    """Teste le système avec des questions de référence"""
    test_questions = [
        "Combien d'élèves y a-t-il à l'ENSAJ?",
        "Quel est l'effectif des étudiants?",
        "Nombre d'étudiants à l'ENSAJ"
    ]
    
    st.sidebar.subheader("🧪 Tests système")
    for question in test_questions:
        if st.sidebar.button(f"Test: {question}"):
            with st.spinner(f"Test: {question}"):
                response = ask_question(question)
                st.sidebar.write(f"**Q:** {question}")
                st.sidebar.write(f"**R:** {response}")

with st.sidebar:
    st.header("⚙️ Configuration du système")
    
    # Mode debug
    st.session_state.debug_mode = st.checkbox("Mode Debug", value=False)
    
    # Initialisation du système
    if st.session_state.faiss_db is None and st.session_state.initialization_error is None:
        with st.status("Initialisation du système...", expanded=True) as status:
            try:
                # 1. Charger les documents
                st.write("📂 Chargement des documents...")
                docs = load_documents()
                
                if not docs:        
                    raise ValueError("Aucun document chargé. Ajoutez des fichiers .txt ou .csv dans le dossier 'data'.")    
                
                st.write(f"📄 **Documents chargés:** {len(docs)}")
                
                # 2. Découper le texte avec des paramètres optimisés
                st.write("✂️ Découpage du texte...")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,  # Réduit pour mieux capturer les informations
                    chunk_overlap=100,  # Augmenté pour éviter de couper les informations
                    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
                )
                chunks = splitter.create_documents(docs)
                st.write(f"📄 **Nombre de chunks:** {len(chunks)}")
                
                # 3. Embeddings + FAISS
                st.write("🔤 Initialisation des embeddings...")
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                test_embedding = embeddings.embed_query("étudiants ENSAJ effectif")
                st.write(f"✅ **Dimension des embeddings:** {len(test_embedding)}")
                
                st.write("🗂️ Création de la base vectorielle FAISS...")
                st.session_state.faiss_db = FAISS.from_documents(chunks, embeddings)
                st.success("✅ Base FAISS créée!")
                
                # 4. Modèle LLM
                st.write("🧠 Initialisation du modèle...")
                st.session_state.model = ChatOllama(
                    model="mistral",
                    temperature=0.1
                )
                # Test du modèle
                test_response = st.session_state.model.invoke([
                    HumanMessage(content="Test: Bonjour")
                ])
                st.success("✅ Modèle initialisé!")
                
                status.update(label="✅ Système prêt!", state="complete", expanded=False)
                
            except Exception as e:
                error_msg = f"❌ Erreur d'initialisation:\n{str(e)}"
                st.session_state.initialization_error = error_msg
                st.error(error_msg)
                status.update(label="❌ Erreur d'initialisation", state="error")
    
    # Afficher l'erreur d'initialisation si elle existe
    if st.session_state.initialization_error:
        st.error("Le système n'a pas pu s'initialiser. Vérifiez la console pour plus de détails.")
    
    # Statistiques
    st.divider()
    st.subheader("📈 Statistiques")
    if st.session_state.faiss_db:
        st.success("✅ Système RAG opérationnel")
    else:
        st.warning("⏳ Initialisation en cours...")
    st.write(f"💬 Messages: {len(st.session_state.messages)}")
    
    # Boutons d'action
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Rafraîchir"):
            st.session_state.faiss_db = None
            st.session_state.initialization_error = None
            st.rerun()
    
    # Tests système
    if st.session_state.faiss_db:
        test_system()

# Section principale de chat
if st.session_state.faiss_db is not None:
    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Suggestions de questions
    if len(st.session_state.messages) == 0:
        st.info("💡 **Suggestions de questions:** Combien d'élèves à l'ENSAJ? Quelles filières? Informations sur les clubs?")
    
    # Input utilisateur
    if prompt := st.chat_input("Posez votre question sur l'ENSAJ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche dans les documents..."):
                response = ask_question(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    if st.session_state.initialization_error:
        st.error("❌ Le système rencontre des problèmes d'initialisation. Vérifiez les documents dans le dossier 'data'.")
    else:
        st.info("⏳ Initialisation du système en cours...")

# Footer avec informations
st.sidebar.divider()
st.sidebar.caption("🤖 Chatbot ENSAJ RAG v2.0 - Système amélioré")
