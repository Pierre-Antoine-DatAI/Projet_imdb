# streamlit_app/app.py
import streamlit as st
import requests
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Recommandations Films", page_icon="🎬")

# URL de ton API locale
API_URL = "http://localhost:8000"

# Titre de l'app
st.title("🎬 Recommandateur de Films")
st.write("Trouvez des films similaires à celui que vous aimez !")

# Interface utilisateur
col1, col2 = st.columns([3, 1])

with col1:
    film_title = st.text_input("Nom du film :", placeholder="Ex: Inception")

with col2:
    top_n = st.number_input("Nombre de recommandations :", min_value=1, max_value=20, value=10)

# Bouton de recherche
if st.button("🔍 Chercher des recommandations"):
    if film_title:
        with st.spinner("Recherche en cours..."):
            try:
                # Appel à l'API
                response = requests.post(
                    f"{API_URL}/recommendations",
                    json={"title": film_title, "top_n": top_n}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success(f"✅ Recommandations pour '{data['film_recherche']}'")
                    
                    # Afficher les résultats dans un tableau
                    df = pd.DataFrame(data['recommendations'])
                    
                    # Formatter les colonnes pour l'affichage
                    df['similarity'] = df['similarity'].round(3)
                    df['vote_average'] = df['vote_average'].round(1)
                    df['popularity'] = df['popularity'].round(1)
                    df['final_score'] = df['final_score'].round(3)
                    
                    st.dataframe(df, use_container_width=True)
                    
                else:
                    error_detail = response.json().get('detail', 'Erreur inconnue')
                    st.error(f"❌ {error_detail}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à l'API. Vérifiez qu'elle est lancée !")
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
    else:
        st.warning("⚠️ Veuillez entrer le nom d'un film")

# Instructions dans la sidebar
st.sidebar.markdown("## 📋 Instructions")
st.sidebar.markdown("1. Lancez l'API : `uvicorn api.main:app --reload --port 8000`")
st.sidebar.markdown("2. Lancez Streamlit : `streamlit run streamlit_app/app.py`")
st.sidebar.markdown("3. Entrez un nom de film et cliquez sur 'Chercher'")

# Test de connexion API
st.sidebar.markdown("## 🔧 Status API")
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("✅ API connectée")
    else:
        st.sidebar.error("❌ API non disponible")
except:
    st.sidebar.error("❌ API non lancée")

# streamlit run streamlit_app/app.py