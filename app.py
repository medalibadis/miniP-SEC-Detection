import streamlit as st
import joblib
import pandas as pd
import os
from src.data_preprocessing import clean_url
from src.feature_engineering import FeatureExtractor

# Page config
st.set_page_config(page_title="SafeURL AI - Cyber Defense", page_icon="🛡️", layout="wide")

# --- PREMIUM CYBER THEME CSS ---
st.markdown("""
    <style>
    /* Dark Mode Theme */
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
    }
    
    /* Centering Container */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    /* Section Headers */
    .section-header {
        border-bottom: 2px solid #1f6feb;
        padding-bottom: 10px;
        margin-top: 50px;
        margin-bottom: 25px;
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
    }

    /* Gradient Hero Title */
    .hero-title {
        background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    /* Predictor Box */
    .predictor-container {
        background: rgba(31, 111, 235, 0.05);
        border: 1px solid #1f6feb;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 40px;
    }

    /* Hide Sidebar */
    [data-testid="stSidebar"] { display: none; }
    
    /* Custom divider */
    .hr-glow {
        height: 2px;
        background: linear-gradient(90deg, transparent, #1f6feb, transparent);
        margin: 60px 0;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Resource Loading
@st.cache_resource
def load_resources():
    try:
        model_path = os.path.join("models", "best_rf_model.pkl")
        extractor_path = os.path.join("models", "feature_extractor.pkl")
        le_path = os.path.join("models", "label_encoder.pkl")
        
        if all(os.path.exists(p) for p in [model_path, extractor_path, le_path]):
            return joblib.load(model_path), joblib.load(extractor_path), joblib.load(le_path)
        return None, None, None
    except Exception as e:
        st.error(f"Critical Error loading models: {e}")
        return None, None, None

model, extractor, le = load_resources()

# --- HERO SECTION ---
st.markdown("<div class='hero-title'>SafeURL AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.4rem; color: #8b949e; margin-bottom: 20px;'>Défense Intelligente contre les URL Malveillantes</p>", unsafe_allow_html=True)

# --- PREDICTOR ---
st.write("### 🔍 Testeur de Menaces en Direct")
url_input = st.text_input("", placeholder="Entrez l'URL à analyser (ex: http://phishing-scam.net)...")
if st.button("Analyser l'URL"):
    if url_input:
        with st.spinner("Analyse sémantique et structurelle..."):
            if model is not None and extractor is not None:
                # Clean URL before extraction
                cleaned_url = clean_url(url_input)
                features = extractor.transform(pd.Series([cleaned_url]))
                pred_idx = model.predict(features)[0]
                probs = model.predict_proba(features)[0]
                label = le.inverse_transform([pred_idx])[0]
                confidence = probs[pred_idx]
                
                color = "#238636" if label == "benign" else "#da3633"
                st.markdown(f"""
                    <div style="background-color: {color}22; border: 2px solid {color}; padding: 30px; border-radius: 12px; text-align: center; margin-top: 20px;">
                        <h1 style="color: {color}; margin: 0; font-size: 3rem;">{label.upper()}</h1>
                        <p style="font-size: 1.5rem; margin-top: 10px;">Fiabilité du modèle : <b>{confidence*100:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("#### Probabilités par Catégorie")
                st.bar_chart(pd.DataFrame({'Confiance': probs}, index=le.classes_))
            else:
                st.error("Les ressources du modèle n'ont pas pu être chargées. Vérifiez le dossier 'models/'.")
    else: st.warning("Veuillez saisir une URL.")

st.markdown("<hr class='hr-glow'>", unsafe_allow_html=True)

# --- 1. INTRODUCTION ---
st.markdown("<div class='section-header'>1. Introduction & Contexte</div>", unsafe_allow_html=True)
st.write("""
Le paysage de la cybercriminalité évolue rapidement. Ce projet implémente un système de classification multi-classe 
basé sur l'IA pour identifier les URL de type **Phishing**, **Malware**, et **Defacement**.

En utilisant le dataset **ISCX-URL2016** (plus de 650 000 échantillons), nous avons construit une solution capable 
de protéger les utilisateurs finaux contre les vecteurs d'attaque les plus courants sur le web.
""")

# --- 2. METHODOLOGY ---
st.markdown("<div class='section-header'>2. Méthodologie & Pipeline</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.write("#### Prétraitement")
    st.markdown("""
    - **Normalisation** : Conversion en minuscules.
    - **Nettoyage** : Suppression des protocoles redondants.
    - **Encodage** : Transformation des labels via `LabelEncoder`.
    """)
with col2:
    st.write("#### Extraction Hybride")
    st.info("**TF-IDF** : Analyse fréquentielle des caractères.")
    st.success("**Handcrafted** : 11 métriques structurelles (Longueur, HTTPS, Points, etc.)")

# --- 3. MODELS ---
st.markdown("<div class='section-header'>3. Architecture des Modèles</div>", unsafe_allow_html=True)
st.write("Nous avons comparé quatre approches distinctes pour garantir une protection maximale :")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Random Forest", "98.1%", "Best")
m2.metric("XGBoost", "97.4%", "Fast")
m3.metric("LSTM", "96.3%", "Deep")
m4.metric("Log. Reg", "91.7%", "Base")

# --- 4. RESULTS & ANALYSIS ---
st.markdown("<div class='section-header'>4. Résultats & Analyse Globale</div>", unsafe_allow_html=True)
if os.path.exists("results/model_results.csv"):
    res = pd.read_csv("results/model_results.csv").sort_values(by='accuracy', ascending=False)
    st.dataframe(res.style.highlight_max(axis=0, subset=['accuracy']), use_container_width=True)

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.write("#### Importance des Caractéristiques")
    if os.path.exists("plots/feature_importance.png"):
        st.image("plots/feature_importance.png")
with col_p2:
    st.write("#### Matrice de Confusion (Random Forest)")
    if os.path.exists("plots/cm_random_forest.png"):
        st.image("plots/cm_random_forest.png")

# --- 5. GLOSSAIRE DES MENACES ---
st.markdown("<div class='section-header'>5. Glossaire des Classifications</div>", unsafe_allow_html=True)
st_c1, st_c2 = st.columns(2)
with st_c1:
    st.markdown("""
    ### ✅ **Benign (Sain)**
    L'URL est identifiée comme sûre. Elle appartient à un site légitime sans activité suspecte détectée.
    
    ### 🎣 **Phishing (Hameçonnage)**
    Site frauduleux imitant une institution réelle (Banque, PayPal) pour voler vos identifiants ou données bancaires.
    """)
with st_c2:
    st.markdown("""
    ### 🦠 **Malware (Logiciel Malveillant)**
    L'URL mène à un site hébergeant des virus, chevaux de Troie ou ransomwares destinés à infecter votre système.
    
    ### 🖼️ **Defacement (Défiguration)**
    Site piraté dont l'apparence a été modifiée par un attaquant pour afficher un message (souvent politique ou revendicateur).
    """)

# --- 6. CONCLUSION ---
st.markdown("<div class='section-header'>6. Conclusion & Étude de Cas</div>", unsafe_allow_html=True)
st.success("""
**Résultat Final** : Le modèle Random Forest a atteint une précision de **98.09%**.
La combinaison des caractéristiques artisanales et du TF-IDF permet de capturer les nuances les plus subtiles des cyber-attaques.
""")
st.write("---")
st.markdown("<p style='text-align: center; color: #8b949e;'>Projet Réalisé par MIHI, KAID, Sennouci & MAZARI - 2SC IASD 2026</p>", unsafe_allow_html=True)
