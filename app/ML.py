import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from scipy.sparse import hstack, csr_matrix
import ast
import gc

# Charger les données
df = pd.read_csv('C:/Users/pierr/Documents/Python/Projet_imdb/data/clean_final2.csv')

# POIDS CONFIGURABLES
weights = {
    'text': 0.30,        # overview + keywords
    'genres': 0.25,     # genres
    'actors': 0.15,      # acteurs
    'directors': 0.20,  # producteurs
    'numeric': 0.1      # budget
}

# Nettoyer les colonnes texte
df['overview'] = df['overview'].fillna('')
df['keywords'] = df['keywords'].fillna('')
df['actor_name'] = df['actor_name'].fillna('')
df['productor_name'] = df['productor_name'].fillna('')


# Préparer les genres (si c'est une string, la convertir en liste)
def parse_genres(x):
    if pd.isna(x) or x == '':
        return []
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except:
        return [item.strip() for item in str(x).split(',') if item.strip()]

df['genres'] = df['genres'].apply(parse_genres)

# Limiter les acteurs aux 5 premiers (pour économiser la RAM)
df['top_actors'] = df['actor_name'].apply(lambda x: x.split(',')[:5] if x else [])
df['top_actors'] = df['top_actors'].apply(lambda x: [item.strip() for item in x if item.strip()])

# Limiter les réalisateurs aux 5 premiers
df['top_directors'] = df['productor_name'].apply(lambda x: x.split(',')[:5] if x else [])
df['top_directors'] = df['top_directors'].apply(lambda x: [item.strip() for item in x if item.strip()])

features_list = []

# 1. Features textuelles (TF-IDF)
if weights['text'] > 0:
    text_data = df['overview'] + ' ' + df['keywords']
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000, max_df=0.8, min_df=2)
    tfidf_matrix = tfidf.fit_transform(text_data)
    tfidf_matrix = tfidf_matrix * weights['text']  # Appliquer le poids
    features_list.append(tfidf_matrix)
    del text_data
    gc.collect()

# 2. Features des genres
if weights['genres'] > 0:
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = mlb_genres.fit_transform(df['genres'])
    genres_encoded = csr_matrix(genres_encoded * weights['genres'])
    features_list.append(genres_encoded)

# 3. Features des acteurs avec embeddings
if weights['actors'] > 0:
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Convertir les listes d'acteurs en texte
    actors_text = df['top_actors'].apply(lambda x: ' '.join(x) if x else '')
    
    # Vectorisation avec limitation
    actor_vectorizer = CountVectorizer(max_features=200, binary=True)
    actors_encoded = actor_vectorizer.fit_transform(actors_text)
    actors_encoded = csr_matrix(actors_encoded * weights['actors'])
    features_list.append(actors_encoded)

# 4. Features des réalisateurs avec embeddings
if weights['directors'] > 0:
    # Convertir les listes de réalisateurs en texte
    directors_text = df['top_directors'].apply(lambda x: ' '.join(x) if x else '')
    
    # Vectorisation avec limitation
    director_vectorizer = CountVectorizer(max_features=100, binary=True)
    directors_encoded = director_vectorizer.fit_transform(directors_text)
    directors_encoded = csr_matrix(directors_encoded * weights['directors'])
    features_list.append(directors_encoded)

# 5. Features numériques
if weights['numeric'] > 0:
    num_features = df[['budget']].fillna(0)
    scaler = MinMaxScaler()
    num_scaled = scaler.fit_transform(num_features)
    num_scaled = csr_matrix(num_scaled * weights['numeric'])
    features_list.append(num_scaled)

# Combiner toutes les features
combined_features = hstack(features_list)

# Entraîner le modèle
nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
nn_model.fit(combined_features)

# Score de qualité pour boost
quality_features = df[['vote_average', 'popularity', 'vote_count']].fillna(0)
quality_scaler = RobustScaler()                    # RobustScaler
quality_scaled = quality_scaler.fit_transform(quality_features)
quality_score = quality_scaled.mean(axis=1)
quality_score = (quality_score - quality_score.min()) / (quality_score.max() - quality_score.min())

def get_recommendations(title, top_n=10):
    # Trouver l'index du film
    idx = df[df['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return f"Le film '{title}' est introuvable."
    idx = idx[0]
    
    # Rechercher les films similaires
    distances, indices = nn_model.kneighbors(combined_features[idx], n_neighbors=top_n*2)           # attention répétition
    
    # Exclure le film lui-même
    similar_indices = indices.flatten()[1:]
    similarities = 1 - distances.flatten()[1:]
    
    # Créer les résultats
    results = pd.DataFrame({
        'title': df.iloc[similar_indices]['title'].values,
        'similarity': similarities,
        'vote_average': df.iloc[similar_indices]['vote_average'].values,
        'popularity' : df.iloc[similar_indices]['popularity'],
        'quality_score': quality_score[similar_indices]
    })
    
    # Score final : similarité (90%) + qualité (10%)
    results['final_score'] = results['similarity'] * 0.9 + results['quality_score'] * 0.1
    results = results.sort_values('final_score', ascending=False)
    
    return results.head(top_n)[['title', 'similarity', 'vote_average', 'popularity', 'final_score']]

# Test
film_input = input("Entrez le nom d'un film : ")
print(f"\n=== Test avec {film_input} ===")
recommendations = get_recommendations(film_input)
print(recommendations)