# Importation des bibliothèques nécessaires
import streamlit as st  # Framework pour créer des applications web
import sys  # Accès aux fonctionnalités système
import traceback  # Pour le débogage des exceptions
from langchain_ollama import OllamaEmbeddings, ChatOllama  # Intégration avec Ollama pour les embeddings et le chat
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Découpage intelligent du texte
from langchain_community.vectorstores import FAISS  # Base de données vectorielle pour la recherche
from langchain_core.messages import SystemMessage, HumanMessage  # Messages pour structurer les conversations
import pandas as pd  # Manipulation de données tabulaires
import os  # Interactions avec le système de fichiers
import json  # Traitement de données JSON

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chatbot ENSAJ - RAG Amélioré",  # Titre de l'onglet du navigateur
    page_icon="🤖",  # Icône de l'application
    layout="wide"  # Utilisation de toute la largeur de la page
)

# Initialisation des variables de session pour persister l'état entre les interactions
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stocke l'historique de la conversation
if "faiss_db" not in st.session_state:
    st.session_state.faiss_db = None  # Stocke la base de données vectorielle
if "model" not in st.session_state:
    st.session_state.model = None  # Stocke le modèle de langage
if "initialization_error" not in st.session_state:
    st.session_state.initialization_error = None  # Stocke les erreurs d'initialisation
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False  # Active/désactive le mode débogage
if "retrieval_context" not in st.session_state:
    st.session_state.retrieval_context = []  # Stocke le contexte récupéré pour debug

# Interface utilisateur principale
st.title("🤖 Chatbot ENSAJ - Système RAG Amélioré")
st.markdown("Posez vos questions sur l'ENSAJ et obtenez des réponses basées sur les documents disponibles.")

def load_documents():
    """
    Charge et prépare les documents texte et CSV depuis le dossier 'data'
    
    Returns:
        list: Liste des contenus textuels des documents
    """
    docs = []  # Liste pour stocker tous les contenus de documents
    data_dir = "data"  # Dossier contenant les données
    
    # Création du dossier s'il n'existe pas
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.warning(f"⚠️ Dossier '{data_dir}' créé. Ajoutez vos fichiers .txt ou .csv")
        return docs  # Retourne une liste vide si pas de documents
    
    # Parcours de tous les fichiers dans le dossier data
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        try:
            # Traitement des fichiers texte (.txt)
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()  # Lecture et nettoyage du contenu
                    if content:
                        # Conservation de toute la ponctuation (importante pour le sens)
                        docs.append(content)
                        st.success(f"✅ {filename}")  # Confirmation du chargement
                    else:
                        st.warning(f"⚠️ {filename} (vide)")  # Avertissement pour fichier vide
            
            # Traitement des fichiers CSV (.csv)
            elif filename.endswith(".csv"):
                # Lecture du fichier CSV avec pandas
                df = pd.read_csv(file_path, encoding="utf-8")
                
                # Méthode 1: Conversion complète en texte
                csv_full_text = df.to_string()
                
                # Méthode 2: Création d'un résumé structuré
                summary_parts = [f"=== Fichier: {filename} ==="]
                summary_parts.append(f"Colonnes: {', '.join(df.columns)}")  # Liste des colonnes
                summary_parts.append(f"Nombre de lignes: {len(df)}")  # Nombre d'enregistrements
                summary_parts.append("\nDONNÉES COMPLÈTES:\n")
                summary_parts.append(csv_full_text)  # Données complètes
                
                # Combinaison de toutes les parties
                combined_content = "\n".join(summary_parts)
                docs.append(combined_content)
                st.success(f"✅ {filename} ({len(df)} lignes)")  # Confirmation avec stats
        
        except Exception as e:
            st.error(f"❌ {filename}: {str(e)}")  # Affichage des erreurs de traitement
    
    return docs  # Retourne tous les documents chargés

def ask_question(query):
    """
    Traite une question en utilisant le système RAG (Retrieval-Augmented Generation)
    
    Args:
        query (str): La question posée par l'utilisateur
        
    Returns:
        str: La réponse générée par le modèle
    """
    try:
        # Vérification que le système est prêt
        if not st.session_state.faiss_db or not st.session_state.model:
            return "❌ Le système RAG n'est pas prêt."
        
        # RÉCUPÉRATION AMÉLIORÉE: Recherche de contexte pertinent
        # Augmentation du nombre de résultats pour plus de contexte
        results = st.session_state.faiss_db.similarity_search(query, k=10)
        
        # Déduplication pour éviter les répétitions
        unique_results = []
        seen_content = set()  # Pour suivre les contenus déjà vus
        for doc in results:
            if doc.page_content.strip() not in seen_content:
                unique_results.append(doc)
                seen_content.add(doc.page_content.strip())
        
        # Affichage debug du contexte récupéré
        if st.session_state.debug_mode:
            st.sidebar.subheader("🔍 Debug - Contexte récupéré")
            for i, doc in enumerate(unique_results[:5]):  # Limité aux 5 premiers
                preview = doc.page_content[:200].replace('\n', ' ')  # Aperçu tronqué
                st.sidebar.write(f"**[{i+1}]** {preview}...")
        
        # Combinaison de tous les contextes pertinents
        context = "\n\n---\n\n".join([doc.page_content for doc in unique_results])
        
        # Sauvegarde pour le débogage (limité aux 500 premiers caractères)
        st.session_state.retrieval_context = context[:500]
        
        # PROMPT AMÉLIORÉ: Instructions strictes pour le modèle
        system_prompt = """Tu es un assistant spécialisé sur l'ENSAJ.

RÈGLES ABSOLUES:
1. Tu DOIS répondre UNIQUEMENT avec les informations du contexte
2. Si une information n'est pas dans le contexte, tu dis: "Les documents ne contiennent pas cette information"
3. Cite TOUJOURS les sources (ex: "D'après le document X...")
4. Pour les nombres/dates/noms, sois EXACTEMENT précis
5. Réponds EN FRANÇAIS
6. Si on demande un nombre et que tu le vois, réponds le nombre EXACTEMENT
7. Ne fais JAMAIS de déductions ou d'hypothèses
8. Cite les portions pertinentes du contexte si nécessaire

Format de réponse:
- Question clairement comprise
- Réponse directe et précise du contexte
- Source/Document d'où vient l'info"""
        
        # Construction des messages pour le modèle
        messages = [
            SystemMessage(content=system_prompt),  # Instructions système
            HumanMessage(content=f"""CONTEXTE DISPONIBLE:
{context}

QUESTION: {query}

Réponds UNIQUEMENT avec ce que tu trouves dans le contexte ci-dessus.""")
        ]
        
        # Appel au modèle pour générer la réponse
        response = st.session_state.model.invoke(messages)
        return response.content  # Retourne le contenu de la réponse
        
    except Exception as e:
        # Gestion des erreurs avec message d'information
        return f"❌ Erreur: {str(e)}\n\nVérifiez que Ollama est lancé et que le modèle mistral est téléchargé."

# =============================================================================
# BARRE LATÉRALE - CONFIGURATION ET CONTRÔLES
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration du système")
    
    # Option pour activer le mode débogage
    st.session_state.debug_mode = st.checkbox("🐛 Mode Debug (voir le contexte récupéré)", value=False)
    
    # Initialisation du système (si pas déjà fait)
    if st.session_state.faiss_db is None and st.session_state.initialization_error is None:
        with st.status("Initialisation du système...", expanded=True) as status:
            try:
                # Étape 1: Chargement des documents
                st.write("📂 Chargement des documents...")
                docs = load_documents()
                
                # Vérification qu'il y a des documents
                if not docs:        
                    raise ValueError("❌ Aucun document. Ajoutez des fichiers .txt ou .csv dans le dossier 'data'")    
                
                st.write(f"✅ {len(docs)} document(s) chargé(s)")
                
                # Étape 2: Découpage du texte en chunks
                st.write("✂️ Découpage du texte...")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,  # Taille optimale pour préserver le contexte
                    chunk_overlap=200,  # Chevauchement pour éviter de couper les informations
                    separators=["\n\n", "\n", ". ", " "]  # Ordre de priorité pour la découpe
                )
                chunks = splitter.create_documents(docs)
                st.write(f"✅ {len(chunks)} chunks créés")
                
                # Étape 3: Initialisation des embeddings
                st.write("🔤 Initialisation des embeddings...")
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                # Test pour vérifier que les embeddings fonctionnent
                test_embedding = embeddings.embed_query("ENSAJ étudiants information")
                st.write(f"✅ Dimension: {len(test_embedding)}")
                
                # Étape 4: Création de la base vectorielle FAISS
                st.write("🗂️ Création de la base vectorielle...")
                st.session_state.faiss_db = FAISS.from_documents(chunks, embeddings)
                st.success("✅ Base FAISS créée!")
                
                # Étape 5: Initialisation du modèle de langage
                st.write("🧠 Initialisation du modèle Mistral...")
                st.session_state.model = ChatOllama(
                    model="mistral",
                    temperature=0.1  # Faible température pour des réponses déterministes
                )
                st.success("✅ Modèle prêt!")
                
                status.update(label="✅ Système prêt!", state="complete", expanded=False)
                
            except Exception as e:
                # Gestion des erreurs d'initialisation
                error_msg = f"❌ Erreur:\n{str(e)}"
                st.session_state.initialization_error = error_msg
                st.error(error_msg)
                status.update(label="❌ Erreur", state="error")
    
    # Affichage des erreurs d'initialisation
    if st.session_state.initialization_error:
        st.error(st.session_state.initialization_error)
    
    st.divider()
    st.subheader("📈 État")
    
    # Indicateur d'état du système
    if st.session_state.faiss_db:
        st.success("✅ Système opérationnel")
    else:
        st.warning("⏳ Initialisation...")
    
    st.write(f"💬 Messages: {len(st.session_state.messages)}")
    
    # Boutons de contrôle
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()  # Recharge la page
    
    if st.button("🔄 Réinitialiser le système"):
        st.session_state.faiss_db = None
        st.session_state.initialization_error = None
        st.rerun()  # Recharge la page
    
    # Affichage du contexte en mode debug
    if st.session_state.debug_mode and st.session_state.retrieval_context:
        st.divider()
        st.subheader("📝 Dernier contexte")
        st.text_area("Contexte:", value=st.session_state.retrieval_context, height=150, disabled=True)

# =============================================================================
# SECTION PRINCIPALE - INTERFACE DE CHAT
# =============================================================================
if st.session_state.faiss_db is not None:
    # Affichage de l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # Affiche user ou assistant
            st.markdown(message["content"])
    
    # Message d'accueil si première utilisation
    if len(st.session_state.messages) == 0:
        st.info("💡 Posez vos questions sur l'ENSAJ. Le chatbot récupère les réponses des documents.")
    
    # Saisie de la question par l'utilisateur
    if prompt := st.chat_input("Question..."):
        # Ajout de la question à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Génération et affichage de la réponse
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche..."):  # Indicateur de progression
                response = ask_question(prompt)
            st.markdown(response)
        
        # Ajout de la réponse à l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    # Message d'erreur si le système n'est pas initialisé
    st.error("❌ Le système n'est pas initialisé. Vérifiez les erreurs ci-dessus.")

# Pied de page dans la barre latérale
st.sidebar.divider()
st.sidebar.caption("Chatbot ENSAJ RAG v3.0")