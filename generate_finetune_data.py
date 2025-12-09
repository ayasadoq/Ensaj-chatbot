import os
import json
import re
import pandas as pd # On ajoute pandas pour lire les CSV
from langchain_ollama import ChatOllama

# --- CONFIGURATION ---
DATA_DIR = "data"
OUTPUT_FILE = "dataset_ensaj.json"
MODEL_NAME = "llama3.2"

# Initialisation du modèle local
llm = ChatOllama(model=MODEL_NAME, temperature=0.7)

def clean_json_text(text):
    """Nettoie la réponse de l'IA pour ne garder que le JSON"""
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def generate_qa(text_chunk, source_type="texte"):
    """Demande à l'IA de créer des Q&A"""
    
    # On adapte un peu le prompt si c'est un tableau (CSV)
    consigne_extra = ""
    if source_type == "csv":
        consigne_extra = "Ce texte provient d'un tableau (CSV). Pose des questions précises sur les données (horaires, noms, emails)."

    prompt = f"""
    Tu es un expert chargé de créer des données d'entraînement.
    
    TA MISSION :
    Lis le contenu ci-dessous ({source_type}) et invente 2 à 3 paires de "Instruction" (Question) et "Output" (Réponse).
    {consigne_extra}
    
    STYLE OBLIGATOIRE :
    - Question d'un étudiant curieux.
    - Réponse d'un étudiant de l'ENSAJ (style naturel, direct).
    
    FORMAT JSON STRICT :
    [
        {{"instruction": "Question...", "input": "", "output": "Réponse..."}},
        {{"instruction": "Question...", "input": "", "output": "Réponse..."}}
    ]

    CONTENU SOURCE :
    {text_chunk}
    
    Génère uniquement le JSON.
    """
    try:
        response = llm.invoke(prompt)
        cleaned = clean_json_text(response.content)
        return json.loads(cleaned)
    except Exception as e:
        # On ignore silencieusement les erreurs de parsing pour ne pas polluer le terminal
        return []

def main():
    print(f"🚀 Démarrage de la génération V2 (TXT + CSV)...")
    all_data = []

    if not os.path.exists(DATA_DIR):
        print(f"❌ Erreur : Dossier '{DATA_DIR}' introuvable.")
        return

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        content = ""
        source_type = "texte"

        try:
            # 1. Traitement des fichiers TXT
            if filename.endswith(".txt"):
                print(f"📄 Lecture de {filename}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            # 2. Traitement des fichiers CSV (NOUVEAU)
            elif filename.endswith(".csv"):
                print(f"📊 Lecture de {filename}...")
                df = pd.read_csv(file_path)
                # On convertit le tableau en texte pour que l'IA puisse le lire
                content = df.to_string(index=False)
                source_type = "csv"
            
            else:
                continue # On ignore les autres fichiers

            # Découpage et Génération
            if content:
                # On découpe en morceaux de 1500 caractères
                chunks = [content[i:i+1500] for i in range(0, len(content), 1500)]
                
                for i, chunk in enumerate(chunks):
                    print(f"   ↳ Génération bloc {i+1}/{len(chunks)}...")
                    pairs = generate_qa(chunk, source_type)
                    if pairs:
                        all_data.extend(pairs)
        
        except Exception as e:
            print(f"⚠️ Erreur lecture fichier {filename}: {e}")

    # Sauvegarde
    print(f"\n💾 Sauvegarde de {len(all_data)} exemples dans {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    
    print("✅ PHASE 1 TERMINÉE ! Vous pouvez vérifier le fichier JSON.")

if __name__ == "__main__":
    main()