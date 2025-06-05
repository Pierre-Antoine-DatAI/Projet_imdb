import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import ast
import gc
from urllib.parse import quote, unquote

# Configuration de la page
st.set_page_config(
    page_title="AlgoCiné - Recommandations de films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .film-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .film-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .film-title {
        color: #FFD700;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 0.5rem;
    }
    
    .film-details {
        color: #E0E0E0;
        font-size: 0.9em;
        line-height: 1.4;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .similarity-bar {
        background: linear-gradient(90deg, #4CAF50 0%, #FFC107 50%, #FF5722 100%);
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    h1, h2, h3 {
        color: white !important;
    }
    
    .stSelectbox label, .stTextInput label, .stSlider label {
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Chargement des données avec gestion d'erreurs
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("clean_final2.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Fichier 'clean_final2.csv' introuvable. Veuillez vérifier le chemin.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données : {str(e)}")
        return pd.DataFrame()

# Initialisation et préparation du modèle
@st.cache_resource
def prepare_model(df):
    if df.empty:
        return None, None, None
    
    # Configuration des poids
    weights = {
        'text': 0.30,
        'genres': 0.25,
        'actors': 0.15,
        'directors': 0.20,
        'numeric': 0.1
    }
    
    # Nettoyage des données
    df['overview'] = df['overview'].fillna('')
    df['keywords'] = df['keywords'].fillna('')
    df['actor_name'] = df['actor_name'].fillna('')
    df['productor_name'] = df['productor_name'].fillna('')
    
    # Préparation des genres
    def parse_genres(x):
        if pd.isna(x) or x == '':
            return []
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except:
            return [item.strip() for item in str(x).split(',') if item.strip()]
    
    df['genres'] = df['genres'].apply(parse_genres)
    
    # Limitation des acteurs et réalisateurs
    df['top_actors'] = df['actor_name'].apply(lambda x: x.split(',')[:5] if x else [])
    df['top_actors'] = df['top_actors'].apply(lambda x: [item.strip() for item in x if item.strip()])
    
    df['top_directors'] = df['productor_name'].apply(lambda x: x.split(',')[:5] if x else [])
    df['top_directors'] = df['top_directors'].apply(lambda x: [item.strip() for item in x if item.strip()])
    
    features_list = []
    
    # Features textuelles
    if weights['text'] > 0:
        text_data = df['overview'] + ' ' + df['keywords']
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000, max_df=0.8, min_df=2)
        tfidf_matrix = tfidf.fit_transform(text_data)
        tfidf_matrix = tfidf_matrix * weights['text']
        features_list.append(tfidf_matrix)
    
    # Features des genres
    if weights['genres'] > 0:
        mlb_genres = MultiLabelBinarizer()
        genres_encoded = mlb_genres.fit_transform(df['genres'])
        genres_encoded = csr_matrix(genres_encoded * weights['genres'])
        features_list.append(genres_encoded)
    
    # Features des acteurs
    if weights['actors'] > 0:
        actors_text = df['top_actors'].apply(lambda x: ' '.join(x) if x else '')
        actor_vectorizer = CountVectorizer(max_features=200, binary=True)
        actors_encoded = actor_vectorizer.fit_transform(actors_text)
        actors_encoded = csr_matrix(actors_encoded * weights['actors'])
        features_list.append(actors_encoded)
    
    # Features des réalisateurs
    if weights['directors'] > 0:
        directors_text = df['top_directors'].apply(lambda x: ' '.join(x) if x else '')
        director_vectorizer = CountVectorizer(max_features=100, binary=True)
        directors_encoded = director_vectorizer.fit_transform(directors_text)
        directors_encoded = csr_matrix(directors_encoded * weights['directors'])
        features_list.append(directors_encoded)
    
    # Features numériques
    if weights['numeric'] > 0:
        num_features = df[['budget']].fillna(0)
        scaler = MinMaxScaler()
        num_scaled = scaler.fit_transform(num_features)
        num_scaled = csr_matrix(num_scaled * weights['numeric'])
        features_list.append(num_scaled)
    
    # Combinaison des features
    combined_features = hstack(features_list)
    
    # Entraînement du modèle
    nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    nn_model.fit(combined_features)
    
    # Score de qualité
    quality_features = df[['vote_average', 'popularity', 'vote_count']].fillna(0)
    quality_scaler = MinMaxScaler()
    quality_scaled = quality_scaler.fit_transform(quality_features)
    quality_score = quality_scaled.mean(axis=1)
    quality_score = (quality_score - quality_score.min()) / (quality_score.max() - quality_score.min())
    
    return nn_model, combined_features, quality_score

def get_recommendations(df, nn_model, combined_features, quality_score, title, top_n=10):
    """Fonction de recommandation améliorée"""
    idx = df[df['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return None
    
    idx = idx[0]
    distances, indices = nn_model.kneighbors(combined_features[idx], n_neighbors=top_n*2)
    similar_indices = indices.flatten()[1:]
    similarities = 1 - distances.flatten()[1:]
    
    results = pd.DataFrame({
        'title': df.iloc[similar_indices]['title'].values,
        'similarity': similarities,
        'vote_average': df.iloc[similar_indices]['vote_average'].values,
        'popularity': df.iloc[similar_indices]['popularity'].values,
        'poster_path': df.iloc[similar_indices]['poster_path'].values if 'poster_path' in df.columns else [''] * len(similar_indices),
        'overview': df.iloc[similar_indices]['overview'].values,
        'actor_name': df.iloc[similar_indices]['actor_name'].values,
        'productor_name': df.iloc[similar_indices]['productor_name'].values,
        'production_companies': df.iloc[similar_indices]['production_companies'].values if 'production_companies' in df.columns else [''] * len(similar_indices),
        'genres': df.iloc[similar_indices]['genres'].values if 'genres' in df.columns else [''] * len(similar_indices)
    })
    
    results['quality_score'] = quality_score[similar_indices]
    results['final_score'] = results['similarity'] * 0.9 + results['quality_score'] * 0.1
    results = results.sort_values('final_score', ascending=False)
    
    return results.head(top_n)

def display_film_card(film_data, index):
    """Affiche une carte de film stylisée avec backdrop"""
    similarity_percent = int(film_data['similarity'] * 100)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Affichage du backdrop - url = f"https://image.tmdb.org/t/p/w500/{df[df['title'] == choix]["poster_path"].values[0]}"
        if pd.notna(film_data['poster_path']) and film_data['poster_path']:
            try:
                st.image(film_data['poster_path'], width=150, caption="")
            except:
                st.markdown("🎬", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="width: 150px; height: 85px; background: rgba(255,255,255,0.1); 
                        display: flex; align-items: center; justify-content: center; 
                        border-radius: 10px; font-size: 2em;">
                🎬
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="film-card">
            <div class="film-title">🎬 {film_data['title']}</div>
            <div class="film-details">
                <strong>📊 Similarité:</strong> {similarity_percent}%<br>
                <div class="similarity-bar" style="width: {similarity_percent}%;"></div>
                <strong>⭐ Note:</strong> {film_data['vote_average']:.1f}/10<br>
                <strong>📈 Popularité:</strong> {film_data['popularity']:.0f}<br>
                <strong>🎭 Acteurs:</strong> {film_data['actor_name'][:80]}{'...' if len(str(film_data['actor_name'])) > 80 else ''}<br>
                <strong>🎬 Réalisateur:</strong> {film_data['productor_name'][:40]}{'...' if len(str(film_data['productor_name'])) > 40 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton pour voir les détails
        if st.button(f"Voir détails", key=f"detail_{index}"):
            st.session_state.selected_film = film_data['title']

def main():
    # Titre principal
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🎬 AlgoCiné</h1>
        <p style="font-size: 1.2em; color: #E0E0E0;">Découvrez votre prochain film préféré grâce à l'intelligence artificielle</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données
    df = load_data()
    if df.empty:
        return
    
    # Préparation du modèle
    with st.spinner("🔄 Préparation du modèle de recommandation..."):
        nn_model, combined_features, quality_score = prepare_model(df)
    
    if nn_model is None:
        st.error("❌ Impossible de préparer le modèle.")
        return
    
    # Sidebar avec options
    with st.sidebar:
        st.markdown("## ⚙️ Options")
        
        # Sélection du nombre de recommandations
        nb_recommendations = st.slider(
            "Nombre de recommandations",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        # Filtre par genre (si disponible)
        if 'genres' in df.columns:
            all_genres = set()
            for genres_list in df['genres'].dropna():
                if isinstance(genres_list, list):
                    all_genres.update(genres_list)
            
            if all_genres:
                selected_genres = st.multiselect(
                    "Filtrer par genre",
                    sorted(list(all_genres)),
                    default=[]
                )
        
        # Filtre par note minimale
        min_rating = st.slider(
            "Note minimale",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )
    
    # Interface principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Recherche de film
        st.markdown("### 🔍 Recherche de film")
        
        # Autocomplete avec liste des films
        film_titles = df['title'].tolist()
        film_choisi = st.selectbox(
            "Sélectionnez ou tapez le nom d'un film :",
            options=[""] + sorted(film_titles),
            index=0,
            help="Commencez à taper pour filtrer les résultats"
        )
        
        # Alternative avec input text
        film_input = st.text_input(
            "Ou saisissez directement le nom :",
            placeholder="Ex: The Matrix, Inception, Avatar..."
        )
        
        # Utiliser l'input text si fourni, sinon le selectbox
        film_final = film_input.strip() if film_input.strip() else film_choisi
    
    with col2:
        # Statistiques du dataset
        st.markdown("### 📊 Base de données")
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df):,}</h3>
            <p>Films disponibles</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Bouton de recherche
    if st.button("🎯 Obtenir des recommandations", type="primary") and film_final:
        with st.spinner(f"🔍 Recherche de films similaires à '{film_final}'..."):
            recommendations = get_recommendations(
                df, nn_model, combined_features, quality_score, 
                film_final, nb_recommendations
            )
        
        if recommendations is None:
            st.error(f"❌ Le film '{film_final}' n'a pas été trouvé dans notre base de données.")
            
            # Suggestions de films similaires
            suggestions = df[df['title'].str.contains(film_final, case=False, na=False)]['title'].head(5)
            if not suggestions.empty:
                st.info("💡 Films similaires disponibles :")
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")
        else:
            # Affichage du film recherché avec poster
            film_original = df[df['title'].str.lower() == film_final.lower()].iloc[0]
            
            st.markdown("---")
            st.markdown(f"### 🎬 Film sélectionné : {film_original['title']}")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Backdrop du film original
                if pd.notna(film_original['poster_path']) and film_original['poster_path']:
                    try:
                        st.image(film_original['poster_path'], width=200, caption="")
                    except:
                        st.markdown("🎬 Image non disponible")
                else:
                    st.markdown("""
                    <div style="width: 200px; height: 110px; background: rgba(255,255,255,0.1); 
                                display: flex; align-items: center; justify-content: center; 
                                border-radius: 10px; font-size: 3em;">
                        🎬
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if film_original['overview']:
                    st.markdown(f"**📝 Synopsis :** {film_original['overview'][:300]}{'...' if len(film_original['overview']) > 300 else ''}")
                st.markdown(f"**🎭 Acteurs :** {film_original['actor_name']}")
                st.markdown(f"**🎬 Réalisateur :** {film_original['productor_name']}")
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>⭐ Note :</strong> {film_original['vote_average']:.1f}/10<br>
                    <strong>📈 Popularité :</strong> {film_original['popularity']:.0f}<br>
                    <strong>🗳️ Votes :</strong> {film_original.get('vote_count', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
            
            # Affichage des recommandations
            st.markdown("---")
            st.markdown(f"### 🎯 {len(recommendations)} films recommandés")
            
            # Appliquer les filtres
            filtered_recommendations = recommendations.copy()
            
            # Filtre par note
            if min_rating > 0:
                filtered_recommendations = filtered_recommendations[
                    filtered_recommendations['vote_average'] >= min_rating
                ]
            
            # Filtre par genre (si sélectionné)
            if 'selected_genres' in locals() and selected_genres:
                mask = filtered_recommendations['genres'].apply(
                    lambda x: any(genre in (x if isinstance(x, list) else []) for genre in selected_genres)
                )
                filtered_recommendations = filtered_recommendations[mask]
            
            if filtered_recommendations.empty:
                st.warning("⚠️ Aucun film ne correspond aux filtres sélectionnés.")
            else:
                # Affichage en grille avec posters
                st.markdown("### 🎯 Recommandations avec aperçu visuel")
                
                # Affichage en grille 3 colonnes pour une meilleure présentation
                for i in range(0, len(filtered_recommendations), 3):
                    cols = st.columns(3)
                    for j, (_, film) in enumerate(filtered_recommendations.iloc[i:i+3].iterrows()):
                        with cols[j]:
                            # Backdrop en petit
                            if pd.notna(film['poster_path']) and film['poster_path']:
                                try:
                                    st.image(film['poster_path'], width=120, caption="")
                                except:
                                    st.markdown("🎬", unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="width: 120px; height: 68px; background: rgba(255,255,255,0.1); 
                                            display: flex; align-items: center; justify-content: center; 
                                            border-radius: 8px; font-size: 1.5em; margin: 0 auto;">
                                    🎬
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Informations du film
                            similarity_percent = int(film['similarity'] * 100)
                            st.markdown(f"""
                            <div style="text-align: center; margin-top: 10px;">
                                <strong style="color: #FFD700;">{film['title']}</strong><br>
                                <span style="color: #4CAF50;">📊 {similarity_percent}%</span><br>
                                <span style="color: #E0E0E0;">⭐ {film['vote_average']:.1f}/10</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Bouton détails
                            if st.button(f"📋 Détails", key=f"detail_grid_{i}_{j}"):
                                st.session_state.selected_film = film['title']
                                st.experimental_rerun()
                
                # Séparateur pour les détails complets
                st.markdown("---")
                st.markdown("### 📋 Détails complets des recommandations")
                
                # Affichage détaillé en liste
                for i, (_, film) in enumerate(filtered_recommendations.iterrows()):
                    display_film_card(film, f"detailed_{i}")
                    st.markdown("<br>", unsafe_allow_html=True)
    
    # Affichage des détails d'un film sélectionné
    if hasattr(st.session_state, 'selected_film'):
        film_details = df[df['title'] == st.session_state.selected_film].iloc[0]
        
        st.markdown("---")
        st.markdown(f"## 🎞️ Détails : {film_details['title']}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Affichage du backdrop
            if pd.notna(film_details['poster_path']) and film_details['poster_path']:
                try:
                    st.image(film_details['poster_path'], width=300, caption="")
                except:
                    st.markdown("""
                    <div style="width: 300px; height: 170px; background: rgba(255,255,255,0.1); 
                                display: flex; align-items: center; justify-content: center; 
                                border-radius: 15px; font-size: 4em;">
                        🎬
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="width: 300px; height: 170px; background: rgba(255,255,255,0.1); 
                            display: flex; align-items: center; justify-content: center; 
                            border-radius: 15px; font-size: 4em;">
                    🎬
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**📝 Synopsis :** {film_details['overview']}")
            st.markdown(f"**🎭 Acteurs :** {film_details['actor_name']}")
            st.markdown(f"**🎬 Réalisateur :** {film_details['productor_name']}")
            if 'production_companies' in film_details:
                st.markdown(f"**🏢 Studio :** {film_details['production_companies']}")
            if 'genres' in film_details and isinstance(film_details['genres'], list):
                st.markdown(f"**🎪 Genres :** {', '.join(film_details['genres'])}")
        
        if st.button("❌ Fermer les détails"):
            del st.session_state.selected_film
            st.experimental_rerun()

if __name__ == "__main__":
    main()

# cd "C:\Users\pierr\Documents\Projet 2\Project"
# streamlit run app.py