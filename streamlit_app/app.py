import streamlit as st
import requests
import pandas as pd
import os

# Configuration de la page
st.set_page_config(
    page_title="Moteur de recommandation de films",
    page_icon="🎬",
    layout="wide"
)

# Styles CSS améliorés
page_element = """
<style>
[data-testid="stAppViewContainer"]{
    background-color: #FFFFFF;
    color: #333333;
}

header, [data-testid="stHeader"]{
    background-color: #FFF200;
}

html, body, [data-testid="stApp"] {
    font-size: 1.2rem;
}

h1, h2, h3, h4, h5, h6 {
    color: #333333;
}

.film-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}

.info-section {
    background: #f1f3f4;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)

# Gestion du logo
logo_path = "Logo_algocine.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=500)
else:
    st.markdown("# 🎬 AlgoCiné")

# Configuration de l'API
API_URL = "http://localhost:8000"

# Headers TMDB pour les bandes-annonces
TMDB_HEADERS = {
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5M2RmN2VkNjI0NjFiOGRjMzg1OGVmZjg4Y2ZiNDU3OCIsIm5iZiI6MTc0NzA1ODQwMy45ODQsInN1YiI6IjY4MjFmZWUzYmNiNmMxODYxNTZlZTQ3YyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.P-D7UIZ_MMXsicKbjbGQtQHb8TNmrF781jVV4DlA7Ws'
}

def get_trailer_url(imdb_id):
    """Récupérer l'URL de la bande-annonce depuis TMDB"""
    if pd.isna(imdb_id) or not imdb_id:
        return None    
    
    url = f"https://api.themoviedb.org/3/movie/{imdb_id}/videos"
    
    try:
        response = requests.get(url, headers=TMDB_HEADERS, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            videos = data.get('results', [])            
            for video in videos:
                if (video.get('type', '').lower() == 'trailer' and 
                    video.get('site', '').lower() == 'youtube'):
                    youtube_key = video.get('key')
                    if youtube_key:
                        return f"https://www.youtube.com/watch?v={youtube_key}"
    except:
        pass
                
    return None

def get_poster_url(poster_path):
    """Construire l'URL du poster depuis TMDB"""
    if pd.notna(poster_path) and poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return None

def afficher_film_selectionne(film_data):
    """Afficher les informations du film sélectionné"""
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.success(f"**Film sélectionné : {film_data.get('title_fr', 'N/A')}**")
    st.markdown('</div>', unsafe_allow_html=True)

    col_poster, col_info = st.columns([1, 3])
    
    with col_poster:
        poster_url = get_poster_url(film_data.get('poster_path'))
        if poster_url:
            st.image(poster_url, width=200)
        else:
            st.write("🎬 Pas d'affiche disponible")
    
    with col_info:
        # Description
        if film_data.get('overview') and film_data['overview'] != 'Pas de description':
            st.markdown(f"*{film_data['overview']}*")
        
        # Informations en colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if film_data.get('release_date'):
                annee = film_data['release_date'][:4] if len(str(film_data['release_date'])) >= 4 else film_data['release_date']
                st.write(f"**Année :** {annee}")
            if film_data.get('genres'):
                st.write(f"**Genres :** {film_data['genres']}")
        
        with col2:
            if film_data.get('productor_name'):
                st.write(f"**Réalisateur :** {film_data['productor_name']}")
            if film_data.get('name_actor'):
                st.write(f"**Acteurs :** {film_data['name_actor']}")
        
        with col3:
            if film_data.get('vote_average') and film_data['vote_average'] != 0:
                note_text = f"**Note :** {film_data['vote_average']}/10"
                if film_data.get('vote_count') and film_data['vote_count'] != 0:
                    note_text += f" ({film_data['vote_count']} votes)"
                st.write(note_text)
        
        # Liens
        liens_col1, liens_col2 = st.columns(2)
        with liens_col1:
            if film_data.get('imdb_id'):
                trailer_url = get_trailer_url(film_data['imdb_id'])
                if trailer_url:
                    st.markdown(f"🎥 **[Voir la bande-annonce]({trailer_url})**")
        
        with liens_col2:
            if film_data.get('homepage') and pd.notna(film_data.get('homepage')):
                st.markdown(f"🌐 **[Site officiel]({film_data['homepage']})**")

def afficher_recommandation_detaillee(rec, numero):
    """Afficher une recommandation avec tous les détails et style amélioré"""
    st.markdown('<div class="film-card">', unsafe_allow_html=True)
    
    # Titre avec numérotation
    st.markdown(f"### {numero}. {rec.get('title_fr', rec.get('titre', 'Titre inconnu'))}")
    
    col_poster, col_info = st.columns([1, 3])
    
    with col_poster:
        poster_url = get_poster_url(rec.get('poster_path'))
        if poster_url:
            st.image(poster_url, width=180)
        elif rec.get('poster_url'):  # Fallback pour l'ancien format
            st.image(rec['poster_url'], width=180)
        else:
            st.markdown("🎬 **Pas d'affiche**")
    
    with col_info:
        # Score de similarité (si disponible)
        if rec.get('similarity'):
            st.markdown(f"**Similarité :** {rec['similarity']}")
        elif rec.get('score'):
            st.markdown(f"**Score :** {rec['score']}")
        
        # Description
        overview = rec.get('overview', rec.get('description', ''))
        if overview and overview != 'Pas de description':
            st.markdown(f"**Synopsis :** {overview}")
        
        # Informations organisées en colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Année
            annee = rec.get('release_date', rec.get('annee', ''))
            if annee:
                if len(str(annee)) >= 4:
                    annee = str(annee)[:4]
                st.write(f"**Année :** {annee}")
            
            # Genres
            genres = rec.get('genres', '')
            if genres and genres != 'N/A':
                st.write(f"**Genres :** {genres}")
        
        with col2:
            # Réalisateur
            realisateur = rec.get('productor_name', rec.get('realisateur', ''))
            if realisateur and realisateur != 'N/A':
                st.write(f"**Réalisateur :** {realisateur}")
            
            # Acteurs
            acteurs = rec.get('name_actor', rec.get('acteurs', ''))
            if acteurs and acteurs != 'N/A':
                # Limiter l'affichage des acteurs pour éviter les textes trop longs
                if len(str(acteurs)) > 50:
                    acteurs = str(acteurs)[:50] + "..."
                st.write(f"**Acteurs :** {acteurs}")
        
        with col3:
            # Note
            note = rec.get('vote_average', rec.get('note', ''))
            if note and note != 'N/A' and note != 0:
                note_text = f"**Note :** {note}/10"
                
                # Nombre de votes
                votes = rec.get('vote_count', rec.get('nb_votes', ''))
                if votes and votes != 'N/A' and votes != 0:
                    note_text += f" ({votes} votes)"
                st.write(note_text)
        
        # Liens (bande-annonce et site officiel)
        st.markdown("---")
        liens_col1, liens_col2 = st.columns(2)
        
        with liens_col1:
            imdb_id = rec.get('imdb_id')
            if imdb_id:
                trailer_url = get_trailer_url(imdb_id)
                if trailer_url:
                    st.markdown(f"🎥 **[Bande-annonce]({trailer_url})**")
                else:
                    st.caption("🎥 Bande-annonce non disponible")
        
        with liens_col2:
            homepage = rec.get('homepage')
            if homepage and pd.notna(homepage):
                st.markdown(f"🌐 **[Site officiel]({homepage})**")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

# Interface utilisateur principale
st.title("🎬 Moteur de Recommandation de Films")
st.markdown("**Trouvez des films similaires à celui que vous aimez !**")

# Interface de saisie améliorée
st.markdown('<div class="info-section">', unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])

with col1:
    film_title = st.text_input(
        "🔍 Nom du film :", 
        placeholder="Ex: Inception, Titanic, Avatar...",
        help="Tapez le nom du film pour lequel vous voulez des recommandations"
    )

with col2:
    top_n = st.number_input(
        "📊 Nombre de recommandations :", 
        min_value=1, 
        max_value=20, 
        value=10,
        help="Entre 1 et 20 recommandations"
    )

# Options d'affichage
affichage_mode = st.radio(
    "🎨 Mode d'affichage :",
    ["Affichage détaillé avec posters", "Tableau simple"],
    horizontal=True,
    help="Choisissez comment afficher les résultats"
)
st.markdown('</div>', unsafe_allow_html=True)

# Bouton de recherche stylé
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    rechercher = st.button("🔍 Obtenir des recommandations", type="primary", use_container_width=True)

# Traitement de la recherche
if rechercher:
    if film_title:
        with st.spinner("🎬 Recherche de recommandations en cours..."):
            try:
                # Appel à l'API
                response = requests.post(
                    f"{API_URL}/recommendations",
                    json={"title": film_title, "top_n": top_n},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Film recherché (si disponible dans la réponse)
                    if 'film_trouve' in data:
                        st.markdown("## 🎯 Film sélectionné")
                        afficher_film_selectionne(data['film_trouve'])
                        st.markdown("---")
                    
                    # Informations sur les résultats
                    st.success(f"✅ Recommandations pour '{data['film_recherche']}'")
                    st.info(f"📊 Nombre de résultats: {data['nombre_resultats']}")
                    
                    recommendations = data['recommendations']
                    
                    if affichage_mode == "Tableau simple":
                        # Affichage en tableau
                        st.markdown("## 📋 Résultats en tableau")
                        df = pd.DataFrame(recommendations)
                        
                        # Formatter les colonnes pour l'affichage
                        if 'similarity' in df.columns:
                            df['similarity'] = df['similarity'].round(3)
                        if 'vote_average' in df.columns:
                            df['vote_average'] = df['vote_average'].round(1)
                        if 'popularity' in df.columns:
                            df['popularity'] = df['popularity'].round(1)
                        if 'final_score' in df.columns:
                            df['final_score'] = df['final_score'].round(3)
                        
                        st.dataframe(df, use_container_width=True)
                    
                    else:
                        # Affichage détaillé avec posters
                        st.markdown("## 🎬 Recommandations détaillées")
                        for i, rec in enumerate(recommendations, 1):
                            afficher_recommandation_detaillee(rec, i)
                    
                elif response.status_code == 404:
                    st.error("❌ Film non trouvé dans la base de données")
                    st.info("💡 Vérifiez l'orthographe ou essayez un autre titre")
                else:
                    error_detail = response.json().get('detail', 'Erreur inconnue')
                    st.error(f"❌ {error_detail}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à l'API")
                st.info("🔧 Vérifiez que l'API est lancée sur le port 8000")
                st.code("uvicorn api.main:app --reload --port 8000")
            except requests.exceptions.Timeout:
                st.error("❌ Timeout: L'API met trop de temps à répondre")
            except Exception as e:
                st.error(f"❌ Erreur inattendue: {str(e)}")
    else:
        st.warning("⚠️ Veuillez entrer le nom d'un film")

# Sidebar avec informations
st.sidebar.markdown("## 📋 Instructions")
st.sidebar.markdown("### 1. **Lancez l'API :**")
st.sidebar.code("uvicorn api.main:app --reload --port 8000")
st.sidebar.markdown("### 2. **Lancez Streamlit :**")
st.sidebar.code("streamlit run streamlit_app/app.py")
st.sidebar.markdown("### 3. **Utilisation :**")
st.sidebar.markdown("- Entrez un nom de film")
st.sidebar.markdown("- Choisissez le nombre de recommandations")
st.sidebar.markdown("- Sélectionnez le mode d'affichage")
st.sidebar.markdown("- Cliquez sur 'Obtenir des recommandations'")

# Test de connexion API amélioré
st.sidebar.markdown("## 🔧 Status API")
try:
    health_response = requests.get(f"{API_URL}/health", timeout=3)
    if health_response.status_code == 200:
        st.sidebar.success("✅ API connectée")
        # Afficher la version si disponible
        health_data = health_response.json()
        if 'message' in health_data:
            st.sidebar.caption(health_data['message'])
        if 'timestamp' in health_data:
            st.sidebar.caption(f"Dernière vérification: {health_data['timestamp']}")
    else:
        st.sidebar.error("❌ API non disponible")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ API non lancée")
    st.sidebar.caption("Lancez: uvicorn api.main:app --reload --port 8000")
except:
    st.sidebar.error("❌ Erreur de connexion")

# Informations supplémentaires dans la sidebar
with st.sidebar.expander("ℹ️ Informations techniques"):
    st.write("**URL de l'API:** http://localhost:8000")
    st.write("**Endpoints disponibles:**")
    st.write("- GET /health")
    st.write("- POST /recommendations")
    st.write("- GET /docs (documentation)")

with st.sidebar.expander("🎬 Fonctionnalités"):
    st.write("- **Posters de films** via TMDB")
    st.write("- **Bandes-annonces** YouTube")
    st.write("- **Sites officiels** des films")
    st.write("- **Informations détaillées** (casting, notes, etc.)")
    st.write("- **Deux modes d'affichage** au choix")

# Footer
st.markdown("---")
st.markdown("**🎬 AlgoCiné** - Système de recommandation de films intelligent")
st.caption("Développé avec Streamlit et FastAPI")


# streamlit run streamlit_app/app.py