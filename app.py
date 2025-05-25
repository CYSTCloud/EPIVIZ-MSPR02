#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Flask pour l'API de prédiction des pandémies
Ce script correspond à la Phase 4 du projet: Développement de l'interface
"""

import os
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Initialisation de l'application Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Permet les requêtes cross-origin

# Créer les dossiers nécessaires s'ils n'existent pas
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Chemins vers les dossiers de modèles et de données
MODELS_DIR = 'models'
DATA_DIR = 'data/by_country'
RESULTS_DIR = 'evaluation'

# Cache pour les données et les modèles
models_cache = {}
data_cache = {}

def load_country_data(country):
    """
    Charge les données pour un pays spécifique depuis le cache ou le fichier
    """
    country_code = country.replace(' ', '_').lower()
    
    # Vérifier si les données sont déjà en cache
    if country_code in data_cache:
        return data_cache[country_code]
    
    # Sinon, charger depuis le fichier
    file_path = os.path.join(DATA_DIR, f"{country_code}.csv")
    
    if not os.path.exists(file_path):
        print(f"Données non disponibles pour {country}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Convertir la date en datetime
        if 'date_value' in df.columns:
            df['date_value'] = pd.to_datetime(df['date_value'])
            
        # Trier par date
        if 'date_value' in df.columns:
            df = df.sort_values('date_value')
        
        # Mettre en cache
        data_cache[country_code] = df
        
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données pour {country}: {str(e)}")
        return None

def load_model(country, model_type='gradient_boosting'):
    """
    Charge un modèle spécifique pour un pays depuis le cache ou le fichier
    """
    country_code = country.replace(' ', '_').lower()
    model_key = f"{country_code}_{model_type}"
    
    # Vérifier si le modèle est déjà en cache
    if model_key in models_cache:
        return models_cache[model_key]
    
    # Sinon, charger depuis le fichier
    if model_type == 'lstm':
        model_path = os.path.join(MODELS_DIR, f"{country_code}_lstm.keras")
        scaler_path = os.path.join(MODELS_DIR, f"{country_code}_lstm_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Modèle LSTM non disponible pour {country}")
            return None
        
        try:
            model = keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            
            # Mettre en cache
            models_cache[model_key] = {
                'model': model,
                'scaler': scaler
            }
            
            return models_cache[model_key]
        except Exception as e:
            print(f"Erreur lors du chargement du modèle LSTM pour {country}: {str(e)}")
            return None
    else:
        model_path = os.path.join(MODELS_DIR, f"{country_code}_{model_type}.pkl")
        
        if not os.path.exists(model_path):
            print(f"Modèle {model_type} non disponible pour {country}")
            return None
        
        try:
            model = joblib.load(model_path)
            
            # Mettre en cache
            models_cache[model_key] = model
            
            return model
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {model_type} pour {country}: {str(e)}")
            return None

def get_available_countries():
    """
    Retourne la liste des pays pour lesquels des modèles sont disponibles
    """
    if not os.path.exists(MODELS_DIR):
        return []
    
    # Extraire les pays à partir des noms de fichiers de modèles
    model_files = os.listdir(MODELS_DIR)
    countries = set()
    
    for file in model_files:
        if file.endswith('.pkl') or file.endswith('.keras'):
            parts = file.split('_')
            if len(parts) > 1:
                country = parts[0]
                countries.add(country)
    
    # Convertir les codes de pays en noms lisibles
    country_names = [country.replace('_', ' ').title() for country in countries]
    
    return sorted(country_names)

def get_best_model_for_country(country):
    """
    Détermine le meilleur modèle pour un pays en fonction des résultats d'évaluation
    """
    country_code = country.replace(' ', '_').lower()
    comparison_path = os.path.join(RESULTS_DIR, f"{country_code}_model_comparison.csv")
    
    if os.path.exists(comparison_path):
        try:
            comparison_df = pd.read_csv(comparison_path)
            if not comparison_df.empty:
                # Trier par RMSE (croissant) ou R2 (décroissant)
                if 'RMSE' in comparison_df.columns:
                    comparison_df = comparison_df.sort_values('RMSE')
                elif 'R2' in comparison_df.columns:
                    comparison_df = comparison_df.sort_values('R2', ascending=False)
                
                if 'Model' in comparison_df.columns and len(comparison_df) > 0:
                    return comparison_df.iloc[0]['Model']
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de comparaison: {str(e)}")
    
    # Par défaut, utiliser gradient_boosting
    return 'gradient_boosting'

def prepare_features(df, target_col='new_cases'):
    """
    Prépare les features pour la prédiction
    """
    # Sélectionner les colonnes pertinentes pour l'entraînement
    date_cols = ['date_value']
    id_cols = ['country', 'id_pandemic']
    
    # Features numériques de base
    basic_numeric = ['total_cases', 'total_deaths', 'new_cases', 'new_deaths']
    
    # Features temporelles
    time_features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'quarter']
    
    # Features dérivées
    derived_features = [col for col in df.columns if '_ma7' in col or '_growth_rate' in col or '_normalized' in col]
    
    # Combiner toutes les features disponibles
    all_features = basic_numeric + time_features + derived_features
    available_features = [col for col in all_features if col in df.columns and col != target_col]
    
    # Créer le dataframe de features
    X = df[available_features].copy()
    
    # Gérer les valeurs manquantes
    X = X.fillna(0)
    
    # Gérer les valeurs infinies ou trop grandes
    for col in X.columns:
        # Remplacer les valeurs infinies par des 0
        X[col] = X[col].replace([np.inf, -np.inf], 0)
        
        # Détecter et remplacer les valeurs extrêmes
        # On considère comme extrême toute valeur > 1e9 ou < -1e9
        X[col] = X[col].apply(lambda x: 0 if abs(x) > 1e9 else x)
    
    return X

def create_lstm_sequences(data, seq_length=14):
    """
    Crée des séquences pour la prédiction avec le modèle LSTM
    """
    return np.array(data[-seq_length:]).reshape(1, seq_length, 1)

def predict_next_days(country, days=30, model_type=None):
    """
    Génère des prédictions pour un nombre spécifié de jours à venir
    """
    # Charger les données du pays
    df = load_country_data(country)
    if df is None:
        return {
            'success': False,
            'message': f"Données non disponibles pour {country}"
        }
    
    # Si aucun modèle spécifique n'est fourni, utiliser le meilleur modèle
    if model_type is None:
        model_type = get_best_model_for_country(country)
    
    # Charger le modèle
    model_data = load_model(country, model_type)
    if model_data is None:
        return {
            'success': False,
            'message': f"Modèle {model_type} non disponible pour {country}"
        }
    
    # Obtenir la dernière date
    if 'date_value' in df.columns:
        last_date = df['date_value'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
    else:
        future_dates = [i+1 for i in range(days)]
    
    # Générer des prédictions
    if model_type == 'lstm':
        # Cas spécial pour LSTM
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Préparer les données
        target_col = 'new_cases'
        data = df[target_col].values.reshape(-1, 1)
        data_scaled = scaler.transform(data)
        
        # Utiliser les dernières valeurs pour générer des prédictions
        seq_length = 14  # Même valeur que pour l'entraînement
        last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)
        
        # Générer des prédictions jour par jour
        predictions_scaled = []
        current_seq = last_sequence.copy()
        
        for _ in range(days):
            # Prédire la prochaine valeur
            next_val_scaled = model.predict(current_seq, verbose=0)[0]
            predictions_scaled.append(next_val_scaled[0])
            
            # Mettre à jour la séquence
            current_seq = np.append(current_seq[:, 1:, :], [[next_val_scaled]], axis=1)
        
        # Inverser la normalisation
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    else:
        # Pour les modèles classiques
        model = model_data
        
        # Préparer les features
        X = prepare_features(df)
        
        # Initialiser avec la dernière ligne
        last_row = X.iloc[-1].copy()
        
        # Initialiser les prédictions
        predictions = []
        
        for i in range(days):
            # Créer un DataFrame pour cette prédiction
            pred_X = pd.DataFrame([last_row.copy()])
            
            # Mettre à jour les features temporelles si disponibles
            if 'date_value' in df.columns:
                future_date = future_dates[i]
                
                if 'year' in pred_X.columns:
                    pred_X['year'] = future_date.year
                if 'month' in pred_X.columns:
                    pred_X['month'] = future_date.month
                if 'day' in pred_X.columns:
                    pred_X['day'] = future_date.day
                if 'day_of_week' in pred_X.columns:
                    pred_X['day_of_week'] = future_date.weekday()
                if 'is_weekend' in pred_X.columns:
                    pred_X['is_weekend'] = 1 if future_date.weekday() >= 5 else 0
                if 'quarter' in pred_X.columns:
                    pred_X['quarter'] = (future_date.month - 1) // 3 + 1
            
            # Gérer les valeurs manquantes ou infinies
            pred_X = pred_X.fillna(0)
            for col in pred_X.columns:
                pred_X[col] = pred_X[col].replace([np.inf, -np.inf], 0)
            
            try:
                # Prédire
                pred = model.predict(pred_X)[0]
                
                # Assurer que la prédiction est non négative
                pred = max(0, pred)
                
                predictions.append(pred)
                
                # Mettre à jour last_row pour la prochaine prédiction
                if 'new_cases' in last_row:
                    last_row['new_cases'] = pred
                if 'total_cases' in last_row and 'new_cases' in last_row:
                    last_row['total_cases'] += pred
            except Exception as e:
                print(f"Erreur lors de la prédiction du jour {i+1}: {str(e)}")
                # En cas d'erreur, utiliser la dernière prédiction ou 0
                predictions.append(predictions[-1] if predictions else 0)
    
    # Formater les dates pour la sortie JSON
    formatted_dates = [date.strftime("%Y-%m-%d") if isinstance(date, datetime) else f"Day {date}" for date in future_dates]
    
    # Récupérer les données historiques pour le graphique
    historical_days = 30  # 30 derniers jours
    if len(df) >= historical_days:
        historical_data = df.iloc[-historical_days:]
    else:
        historical_data = df
    
    if 'date_value' in historical_data.columns:
        historical_dates = [date.strftime("%Y-%m-%d") for date in historical_data['date_value']]
    else:
        historical_dates = [f"Day {i+1}" for i in range(len(historical_data))]
    
    historical_cases = historical_data['new_cases'].tolist() if 'new_cases' in historical_data.columns else []
    
    # Nous gardons le code pour générer l'image pour la compatibilité avec les anciennes versions
    # mais nous n'utilisons plus cette image avec la nouvelle interface Chart.js
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Tracer les données historiques
    ax.plot(range(len(historical_cases)), historical_cases, label='Données historiques', color='blue')
    
    # Tracer les prédictions
    ax.plot(range(len(historical_cases), len(historical_cases) + len(predictions)), predictions, 
             label=f'Prédictions ({model_type})', color='red', linestyle='--')
    
    # Ajouter une ligne verticale pour séparer l'historique des prédictions
    ax.axvline(x=len(historical_cases) - 0.5, color='green', linestyle='-', alpha=0.5, 
                label='Limite historique/prédictions')
    
    # Ajouter des labels et une légende
    ax.set_title(f'Prévision des nouveaux cas pour {country} - {days} jours')
    ax.set_xlabel('Jours')
    ax.set_ylabel('Nouveaux cas')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Sauvegarder le graphique dans un buffer
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    
    # Convertir en base64 pour l'affichage HTML
    plot_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    # Résultat
    return {
        'success': True,
        'country': country,
        'model': model_type,
        'days': days,
        'dates': formatted_dates,
        'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        'historical_dates': historical_dates,
        'historical_cases': historical_cases,
        'plot': plot_data
    }


# Routes API

@app.route('/')
def index():
    """Page d'accueil"""
    countries = get_available_countries()
    return render_template('index.html', countries=countries)

@app.route('/api/countries')
def api_countries():
    """Retourne la liste des pays disponibles"""
    countries = get_available_countries()
    return jsonify({
        'success': True,
        'countries': countries
    })

@app.route('/country/<country>')
def country_page(country):
    """Page de détail pour un pays spécifique"""
    # Valider que le pays existe
    countries = get_available_countries()
    if country.title() not in countries:
        return render_template('error.html', message=f"Pays non disponible: {country}")
    
    # Obtenir les modèles disponibles pour ce pays
    country_code = country.replace(' ', '_').lower()
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(country_code) and f.endswith('.pkl')]
    models = [f.replace(f"{country_code}_", "").replace(".pkl", "") for f in model_files]
    
    # Vérifier si un modèle LSTM est disponible
    if os.path.exists(os.path.join(MODELS_DIR, f"{country_code}_lstm.keras")):
        models.append('lstm')
    
    # Obtenir le meilleur modèle
    best_model = get_best_model_for_country(country)
    
    return render_template('country.html', country=country, models=models, best_model=best_model)

@app.route('/api/predict/<country>')
def api_predict(country):
    """API pour obtenir des prédictions pour un pays"""
    # Valider que le pays existe
    countries = get_available_countries()
    if country.title() not in countries:
        return jsonify({
            'success': False,
            'message': f"Pays non disponible: {country}"
        })
    
    # Obtenir les paramètres
    days = request.args.get('days', default=30, type=int)
    model_type = request.args.get('model', default=None, type=str)
    
    # Limiter le nombre de jours pour éviter les abus
    if days > 365:
        days = 365
    
    # Générer les prédictions
    result = predict_next_days(country.title(), days, model_type)
    
    return jsonify(result)

@app.route('/api/historical/<country>')
def api_historical(country):
    """API pour obtenir les données historiques d'un pays"""
    # Valider que le pays existe
    countries = get_available_countries()
    if country.title() not in countries:
        return jsonify({
            'success': False,
            'message': f"Pays non disponible: {country}"
        })
    
    # Charger les données
    df = load_country_data(country.title())
    if df is None:
        return jsonify({
            'success': False,
            'message': f"Données non disponibles pour {country}"
        })
    
    # Limiter à 365 jours pour éviter des réponses trop volumineuses
    days = request.args.get('days', default=365, type=int)
    if days > 365:
        days = 365
    
    if len(df) > days:
        df = df.iloc[-days:]
    
    # Formater les données pour la sortie JSON
    if 'date_value' in df.columns:
        dates = [date.strftime("%Y-%m-%d") for date in df['date_value']]
    else:
        dates = [f"Day {i+1}" for i in range(len(df))]
    
    cases = df['new_cases'].tolist() if 'new_cases' in df.columns else []
    deaths = df['new_deaths'].tolist() if 'new_deaths' in df.columns else []
    
    return jsonify({
        'success': True,
        'country': country.title(),
        'dates': dates,
        'new_cases': cases,
        'new_deaths': deaths
    })

@app.route('/api/compare')
def api_compare():
    """API pour comparer plusieurs pays"""
    # Obtenir les pays à comparer
    countries_param = request.args.get('countries', default='', type=str)
    if not countries_param:
        return jsonify({
            'success': False,
            'message': "Aucun pays spécifié"
        })
    
    countries = countries_param.split(',')
    countries = [country.strip().title() for country in countries]
    
    # Valider que les pays existent
    available_countries = get_available_countries()
    valid_countries = [country for country in countries if country in available_countries]
    
    if not valid_countries:
        return jsonify({
            'success': False,
            'message': "Aucun pays valide spécifié"
        })
    
    # Paramètres
    days = request.args.get('days', default=30, type=int)
    if days > 365:
        days = 365
    
    # Préparer les résultats
    comparison = {
        'success': True,
        'countries': valid_countries,
        'days': days,
        'dates': [],
        'predictions': {}
    }
    
    # Générer des prédictions pour chaque pays
    for country in valid_countries:
        result = predict_next_days(country, days)
        if result['success']:
            comparison['dates'] = result['dates']  # Toutes les dates seront les mêmes
            comparison['predictions'][country] = result['predictions']
    
    return jsonify(comparison)

@app.route('/compare')
def compare_page():
    """Page de comparaison entre pays"""
    countries = get_available_countries()
    return render_template('compare.html', countries=countries)

@app.route('/static/<path:path>')
def send_static(path):
    """Servir les fichiers statiques"""
    return send_from_directory('static', path)


# Création des templates HTML de base

def create_templates():
    """Crée les templates HTML nécessaires s'ils n'existent pas"""
    # Créer le dossier templates s'il n'existe pas
    os.makedirs('templates', exist_ok=True)
    
    # Template de base (layout.html)
    layout_html = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}EPIVIZ - Prédictions de Pandémies{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; }
        .prediction-chart { max-width: 100%; height: auto; }
    </style>
    {% block head %}{% endblock %}
</head>
<body>
    <div class="container">
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
            <div class="container-fluid">
                <a class="navbar-brand" href="/">EPIVIZ</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link" href="/">Accueil</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/compare">Comparaison</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <div class="row">
            <div class="col-12">
                {% block content %}{% endblock %}
            </div>
        </div>
        
        <footer class="mt-5 pt-3 border-top text-center">
            <p>EPIVIZ - Prédictions de Pandémies © 2025</p>
        </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
"""
    
    # Template de la page d'accueil (index.html)
    index_html = """{% extends "layout.html" %}

{% block title %}EPIVIZ - Accueil{% endblock %}

{% block content %}
<div class="jumbotron bg-light p-5 rounded">
    <h1 class="display-4">EPIVIZ - Prédictions de Pandémies</h1>
    <p class="lead">Plateforme d'analyse et de prédiction des données de pandémies utilisant des modèles d'apprentissage automatique avancés.</p>
    <hr class="my-4">
    <p>Sélectionnez un pays pour voir les prédictions :</p>
</div>

<div class="mt-4">
    <h2>Pays disponibles</h2>
    <div class="row row-cols-1 row-cols-md-3 g-4 mt-2">
        {% for country in countries %}
        <div class="col">
            <div class="card h-100">
                <div class="card-body">
                    <h5 class="card-title">{{ country }}</h5>
                    <p class="card-text">Accédez aux prédictions pour {{ country }}</p>
                    <a href="/country/{{ country }}" class="btn btn-primary">Voir les prédictions</a>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</div>
{% endblock %}
"""
    
    # Template pour la page d'un pays (country.html)
    country_html = """{% extends "layout.html" %}

{% block title %}Prédictions pour {{ country }}{% endblock %}

{% block content %}
<h1>Prédictions pour {{ country }}</h1>

<div class="row mb-4">
    <div class="col-md-4">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">Options de prédiction</h5>
            </div>
            <div class="card-body">
                <form id="prediction-form">
                    <div class="mb-3">
                        <label for="days" class="form-label">Nombre de jours à prédire :</label>
                        <input type="number" class="form-control" id="days" name="days" min="7" max="365" value="30">
                    </div>
                    <div class="mb-3">
                        <label for="model" class="form-label">Modèle :</label>
                        <select class="form-select" id="model" name="model">
                            <option value="">Meilleur modèle ({{ best_model }})</option>
                            {% for model in models %}
                            <option value="{{ model }}">{{ model }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Générer les prédictions</button>
                </form>
            </div>
        </div>
    </div>
    <div class="col-md-8">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">Prédictions</h5>
            </div>
            <div class="card-body text-center">
                <div id="loading" class="d-none">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Chargement...</span>
                    </div>
                    <p>Génération des prédictions en cours...</p>
                </div>
                <div id="prediction-result">
                    <p>Ajustez les paramètres et cliquez sur "Générer les prédictions" pour voir les résultats.</p>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="card mb-4 d-none" id="prediction-chart-card">
    <div class="card-header bg-primary text-white">
        <h5 class="mb-0">Graphique de prédiction</h5>
    </div>
    <div class="card-body text-center">
        <img id="prediction-chart" class="img-fluid prediction-chart" src="" alt="Graphique de prédiction">
    </div>
</div>

<div class="card d-none" id="prediction-table-card">
    <div class="card-header bg-primary text-white">
        <h5 class="mb-0">Données de prédiction</h5>
    </div>
    <div class="card-body">
        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Nouveaux cas prédits</th>
                    </tr>
                </thead>
                <tbody id="prediction-table-body">
                </tbody>
            </table>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const loading = document.getElementById('loading');
    const resultDiv = document.getElementById('prediction-result');
    const chartCard = document.getElementById('prediction-chart-card');
    const chartImg = document.getElementById('prediction-chart');
    const tableCard = document.getElementById('prediction-table-card');
    const tableBody = document.getElementById('prediction-table-body');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const days = document.getElementById('days').value;
        const model = document.getElementById('model').value;
        
        // Afficher le chargement
        loading.classList.remove('d-none');
        resultDiv.innerHTML = '';
        chartCard.classList.add('d-none');
        tableCard.classList.add('d-none');
        
        try {
            // Construire l'URL
            let url = `/api/predict/{{ country }}?days=${days}`;
            if (model) {
                url += `&model=${model}`;
            }
            
            // Faire la requête
            const response = await fetch(url);
            const data = await response.json();
            
            if (data.success) {
                // Afficher les résultats
                resultDiv.innerHTML = `<div class="alert alert-success">Prédictions générées avec succès pour ${data.country} en utilisant le modèle ${data.model}.</div>`;
                
                // Afficher le graphique
                chartImg.src = `data:image/png;base64,${data.plot}`;
                chartCard.classList.remove('d-none');
                
                // Remplir le tableau
                tableBody.innerHTML = '';
                for (let i = 0; i < data.dates.length; i++) {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${data.dates[i]}</td>
                        <td>${Math.round(data.predictions[i])}</td>
                    `;
                    tableBody.appendChild(row);
                }
                tableCard.classList.remove('d-none');
            } else {
                resultDiv.innerHTML = `<div class="alert alert-danger">${data.message}</div>`;
            }
        } catch (error) {
            resultDiv.innerHTML = `<div class="alert alert-danger">Erreur lors de la génération des prédictions: ${error.message}</div>`;
        } finally {
            loading.classList.add('d-none');
        }
    });
});
</script>
{% endblock %}
"""
    
    # Template pour la page de comparaison (compare.html)
    compare_html = """{% extends "layout.html" %}

{% block title %}Comparaison des pays{% endblock %}

{% block content %}
<h1>Comparaison des pays</h1>

<div class="row mb-4">
    <div class="col-md-4">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">Options de comparaison</h5>
            </div>
            <div class="card-body">
                <form id="comparison-form">
                    <div class="mb-3">
                        <label class="form-label">Sélectionnez les pays à comparer :</label>
                        <div class="border p-3 rounded" style="max-height: 300px; overflow-y: auto;">
                            {% for country in countries %}
                            <div class="form-check">
                                <input class="form-check-input country-check" type="checkbox" value="{{ country }}" id="country-{{ loop.index }}">
                                <label class="form-check-label" for="country-{{ loop.index }}">
                                    {{ country }}
                                </label>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    <div class="mb-3">
                        <label for="compare-days" class="form-label">Nombre de jours à prédire :</label>
                        <input type="number" class="form-control" id="compare-days" name="days" min="7" max="90" value="30">
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Comparer les pays</button>
                </form>
            </div>
        </div>
    </div>
    <div class="col-md-8">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">Résultats de la comparaison</h5>
            </div>
            <div class="card-body text-center">
                <div id="comparison-loading" class="d-none">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Chargement...</span>
                    </div>
                    <p>Génération de la comparaison en cours...</p>
                </div>
                <div id="comparison-result">
                    <p>Sélectionnez au moins deux pays et cliquez sur "Comparer les pays" pour voir les résultats.</p>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="card mb-4 d-none" id="comparison-chart-card">
    <div class="card-header bg-primary text-white">
        <h5 class="mb-0">Comparaison des prédictions</h5>
    </div>
    <div class="card-body">
        <canvas id="comparison-chart"></canvas>
    </div>
</div>
{% endblock %}

{% block head %}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
{% endblock %}

{% block scripts %}
<script>
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('comparison-form');
    const loading = document.getElementById('comparison-loading');
    const resultDiv = document.getElementById('comparison-result');
    const chartCard = document.getElementById('comparison-chart-card');
    const countryCheckboxes = document.querySelectorAll('.country-check');
    
    let comparisonChart = null;
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Récupérer les pays sélectionnés
        const selectedCountries = [];
        countryCheckboxes.forEach(checkbox => {
            if (checkbox.checked) {
                selectedCountries.push(checkbox.value);
            }
        });
        
        if (selectedCountries.length < 2) {
            resultDiv.innerHTML = `<div class="alert alert-warning">Veuillez sélectionner au moins deux pays à comparer.</div>`;
            return;
        }
        
        const days = document.getElementById('compare-days').value;
        
        // Afficher le chargement
        loading.classList.remove('d-none');
        resultDiv.innerHTML = '';
        chartCard.classList.add('d-none');
        
        try {
            // Construire l'URL
            const url = `/api/compare?countries=${selectedCountries.join(',')}&days=${days}`;
            
            // Faire la requête
            const response = await fetch(url);
            const data = await response.json();
            
            if (data.success) {
                // Afficher les résultats
                resultDiv.innerHTML = `<div class="alert alert-success">Comparaison générée avec succès pour ${data.countries.length} pays.</div>`;
                
                // Préparer les données pour le graphique
                const chartData = {
                    labels: data.dates,
                    datasets: []
                };
                
                // Couleurs pour les lignes du graphique
                const colors = [
                    'rgb(255, 99, 132)',
                    'rgb(54, 162, 235)',
                    'rgb(255, 206, 86)',
                    'rgb(75, 192, 192)',
                    'rgb(153, 102, 255)',
                    'rgb(255, 159, 64)',
                    'rgb(199, 199, 199)'
                ];
                
                // Ajouter chaque pays au graphique
                let colorIndex = 0;
                for (const country of data.countries) {
                    chartData.datasets.push({
                        label: country,
                        data: data.predictions[country],
                        borderColor: colors[colorIndex % colors.length],
                        backgroundColor: 'transparent',
                        tension: 0.1
                    });
                    colorIndex++;
                }
                
                // Afficher le graphique
                chartCard.classList.remove('d-none');
                const ctx = document.getElementById('comparison-chart').getContext('2d');
                
                if (comparisonChart) {
                    comparisonChart.destroy();
                }
                
                comparisonChart = new Chart(ctx, {
                    type: 'line',
                    data: chartData,
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: `Comparaison des prédictions pour ${days} jours`
                            },
                            tooltip: {
                                mode: 'index',
                                intersect: false
                            }
                        },
                        scales: {
                            y: {
                                title: {
                                    display: true,
                                    text: 'Nouveaux cas prédits'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Date'
                                }
                            }
                        }
                    }
                });
            } else {
                resultDiv.innerHTML = `<div class="alert alert-danger">${data.message}</div>`;
            }
        } catch (error) {
            resultDiv.innerHTML = `<div class="alert alert-danger">Erreur lors de la génération de la comparaison: ${error.message}</div>`;
        } finally {
            loading.classList.add('d-none');
        }
    });
});
</script>
{% endblock %}
"""
    
    # Template pour la page d'erreur (error.html)
    error_html = """{% extends "layout.html" %}

{% block title %}Erreur{% endblock %}

{% block content %}
<div class="alert alert-danger">
    <h4 class="alert-heading">Erreur!</h4>
    <p>{{ message }}</p>
    <hr>
    <p class="mb-0"><a href="/" class="alert-link">Retourner à l'accueil</a></p>
</div>
{% endblock %}
"""
    
    # Écrire les templates s'ils n'existent pas
    if not os.path.exists('templates/layout.html'):
        with open('templates/layout.html', 'w', encoding='utf-8') as f:
            f.write(layout_html)
    
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write(index_html)
    
    if not os.path.exists('templates/country.html'):
        with open('templates/country.html', 'w', encoding='utf-8') as f:
            f.write(country_html)
    
    if not os.path.exists('templates/compare.html'):
        with open('templates/compare.html', 'w', encoding='utf-8') as f:
            f.write(compare_html)
    
    if not os.path.exists('templates/error.html'):
        with open('templates/error.html', 'w', encoding='utf-8') as f:
            f.write(error_html)


# Point d'entrée principal

if __name__ == '__main__':
    print("=== DÉMARRAGE DE L'API EPIVIZ ===")
    print(f"Date et heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Créer les templates HTML nécessaires
    create_templates()
    
    # Afficher les pays disponibles
    countries = get_available_countries()
    print(f"Pays disponibles: {', '.join(countries)}")
    
    # Démarrer l'application Flask
    app.run(debug=True, host='0.0.0.0', port=8000)
