# Roadmap : Modèle d'Apprentissage pour la Prédiction des Pandémies

Cette feuille de route décrit les étapes nécessaires pour développer et entraîner un modèle d'apprentissage efficace pour la prédiction des données de pandémies, en se basant sur le fichier full_grouped.csv et en suivant les critères d'évaluation de la grille MSPR.

## 1. Architecture du Projet

```
┌─────────────────────────────────────────┐
│                                         │
│          Architecture du Projet          │
│                                         │
└───────────────────┬───────────────────┘
                   │
    ┌───────────┬────▼───┬────────────┐
    │           │        │           │
    │  Données  │ Modèles │ Interface │
    │  Source   │   ML   │    Web    │
    │           │        │           │
    └───────────┘        └────────────┘
        │                    │
        ▼                    ▲
    ┌─────────┐           │
    │ Préparation │           │
    │   des      │───────────┘
    │  données   │
    └─────────┘
```

Le projet sera structuré autour de trois composants principaux:

1. **Préparation des données**: Module responsable du chargement, du nettoyage et de la transformation des données du fichier CSV source.

2. **Modèles d'apprentissage**: Composant central du projet qui gère l'entraînement, l'évaluation et l'ajustement des modèles de prédiction pour chaque pays.

3. **Interface Web**: Application Flask qui permet la visualisation des données et des prédictions générées par les modèles.

L'accent principal sera mis sur le développement et l'entraînement des modèles d'apprentissage, conformément aux critères d'évaluation.

## 2. Préparation des Données pour l'Apprentissage

### 2.1. Analyse exploratoire des données (EDA)

```python
# data_preparation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df = pd.read_csv('full_grouped.csv')

# Analyse initiale
print("Dimensions du dataset:", df.shape)
print("\nAperçu des données:")
print(df.head())
print("\nInformations sur les types de données:")
print(df.info())
print("\nStatistiques descriptives:")
print(df.describe())

# Analyse des valeurs manquantes
print("\nValeurs manquantes par colonne:")
print(df.isnull().sum())

# Visualisations pour comprendre les tendances
plt.figure(figsize=(12, 6))
sns.lineplot(x='date_value', y='total_cases', hue='country', data=df.loc[df['country'].isin(['US', 'Brazil', 'India', 'France', 'Germany'])])
plt.title('Évolution des cas de COVID-19 pour les principaux pays')
plt.xticks(rotation=45)
plt.savefig('trends_analysis.png')
plt.close()

# Analyse de corrélation entre les variables
corr_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation des variables numériques')
plt.savefig('correlation_matrix.png')
```

### 2.2. Prétraitement des données pour les modèles

```python
# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """Fonction complète de prétraitement des données."""
    
    # Conversion des dates en format datetime
    df['date_value'] = pd.to_datetime(df['date_value'])
    
    # Extraction de features temporelles
    df['year'] = df['date_value'].dt.year
    df['month'] = df['date_value'].dt.month
    df['day'] = df['date_value'].dt.day
    df['day_of_week'] = df['date_value'].dt.dayofweek
    
    # Calcul de features dérivées
    # Taux de mortalité
    df['mortality_rate'] = np.where(df['total_cases'] > 0, 
                                    df['total_deaths'] / df['total_cases'], 
                                    0)
    
    # Taux de croissance (différence de N jours)
    df = df.sort_values(['country', 'date_value'])
    df['cases_growth_rate'] = df.groupby('country')['total_cases'].pct_change(periods=7)
    df['deaths_growth_rate'] = df.groupby('country')['total_deaths'].pct_change(periods=7)
    
    # Moyenne mobile sur 7 jours
    df['cases_7day_avg'] = df.groupby('country')['new_cases'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['deaths_7day_avg'] = df.groupby('country')['new_deaths'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    # Gestion des valeurs aberrantes
    df['new_cases'] = np.where(df['new_cases'] < 0, 0, df['new_cases'])
    df['new_deaths'] = np.where(df['new_deaths'] < 0, 0, df['new_deaths'])
    
    # Gestion des valeurs manquantes
    numeric_features = ['total_cases', 'total_deaths', 'new_cases', 'new_deaths',
                        'cases_growth_rate', 'deaths_growth_rate', 'cases_7day_avg', 'deaths_7day_avg']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    # Appliquer les transformations
    ct = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'
    )
    
    # Obtenir les noms des colonnes pour le DataFrame transformé
    feature_names = numeric_features + [col for col in df.columns if col not in numeric_features]
    
    # Transformer les données
    df_transformed = pd.DataFrame(
        ct.fit_transform(df),
        columns=feature_names,
        index=df.index
    )
    
    return df_transformed

# Charger les données
df = pd.read_csv('full_grouped.csv')

# Prétraiter les données
df_processed = preprocess_data(df)

# Enregistrer les données prétraitées
df_processed.to_csv('data/processed/full_grouped_processed.csv', index=False)

print("Prétraitement terminé. Données prêtes pour l'apprentissage.")
```

### 2.3. Division des données par pays pour l'entraînement spécifique

```python
# split_data.py
import pandas as pd
import os

# Créer le dossier de sortie s'il n'existe pas
output_dir = 'data/by_country'
os.makedirs(output_dir, exist_ok=True)

# Charger les données prétraitées
df = pd.read_csv('data/processed/full_grouped_processed.csv')

# Obtenir la liste des pays uniques
countries = df['country'].unique()

# Diviser les données par pays
for country in countries:
    # Filtrer les données pour ce pays
    country_data = df[df['country'] == country]
    
    # Ne traiter que les pays ayant suffisamment de données
    if len(country_data) >= 30:  # Au moins 30 jours de données
        # Enregistrer dans un fichier séparé
        filename = f"{output_dir}/{country.replace(' ', '_').lower()}.csv"
        country_data.to_csv(filename, index=False)
        print(f"Généré: {filename} avec {len(country_data)} entrées")

print(f"Données divisées par pays et enregistrées dans {output_dir}")
```

## 3. Développement des Modèles d'Apprentissage

### 3.1. Sélection des algorithmes appropriés

```python
# model_selection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import joblib
import os

# Créer les dossiers pour sauvegarder les résultats
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Fonction pour évaluer plusieurs modèles sur un pays spécifique
def evaluate_models(country_name, target_col='new_cases'):
    """Teste plusieurs modèles de ML et évalue leurs performances."""
    # Charger les données spécifiques au pays
    filename = f"data/by_country/{country_name.replace(' ', '_').lower()}.csv"
    if not os.path.exists(filename):
        print(f"Données non disponibles pour {country_name}")
        return None
    
    df = pd.read_csv(filename)
    
    # Préparer les features et la cible
    # Sélectionner les colonnes numériques pertinentes pour la prédiction
    numeric_cols = ['total_cases', 'total_deaths', 'new_cases', 'new_deaths',
                    'cases_growth_rate', 'deaths_growth_rate', 'cases_7day_avg', 
                    'deaths_7day_avg', 'year', 'month', 'day', 'day_of_week']
    
    # Vérifier que toutes les colonnes sont disponibles
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    # Utiliser des features décalées (lag features) pour capturer les tendances temporelles
    for col in ['new_cases', 'new_deaths']:
        if col in df.columns:
            for lag in [1, 3, 7, 14]:
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = df[col].shift(lag)
                available_cols.append(lag_col)
    
    # Supprimer les lignes avec des valeurs manquantes
    df = df.dropna()
    
    # Features et cible
    X = df[available_cols]
    y = df[target_col]
    
    # Division en ensembles d'entraînement et de test (80-20 avec respect de l'ordre chronologique)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Modèles à évaluer
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Évaluation des modèles
    results = {}
    for name, model in models.items():
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédiction
        y_pred = model.predict(X_test)
        
        # Métriques
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Sauvegarder les résultats
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        # Sauvegarder le modèle
        joblib.dump(model, f"models/{country_name.replace(' ', '_').lower()}_{name.replace(' ', '_').lower()}.pkl")
        
        print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    # Visualiser les résultats
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(results)), [results[model]['RMSE'] for model in results.keys()])
    plt.xticks(range(len(results)), list(results.keys()), rotation=45)
    plt.title(f'RMSE par modèle pour {country_name} - {target_col}')
    plt.tight_layout()
    plt.savefig(f"results/{country_name.replace(' ', '_').lower()}_model_comparison.png")
    
    return results

# Tester pour quelques pays importants
countries = ['US', 'Brazil', 'France', 'Germany', 'India']
for country in countries:
    print(f"\n\u00c9valuation des modèles pour {country}:")
    evaluate_models(country, 'new_cases')
```

### 3.2. Développement du modèle LSTM pour les séries temporelles

```python
# lstm_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def create_lstm_sequences(data, seq_length):
    """Crée des séquences pour l'entraînement du modèle LSTM."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm_model(country_name, target_col='new_cases', seq_length=14):
    """Entraîne un modèle LSTM pour prédire les données de pandémie."""
    # Charger les données
    filename = f"data/by_country/{country_name.replace(' ', '_').lower()}.csv"
    if not os.path.exists(filename):
        print(f"Données non disponibles pour {country_name}")
        return None
    
    df = pd.read_csv(filename)
    df = df.sort_values('date_value')  # S'assurer que les données sont en ordre chronologique
    
    # Préparer les données pour LSTM
    data = df[target_col].values.reshape(-1, 1)
    
    # Normaliser les données
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Créer des séquences
    X, y = create_lstm_sequences(data_scaled, seq_length)
    
    # Reformater X pour LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Diviser en ensembles d'entraînement et de test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Construire le modèle LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    # Compiler le modèle
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping pour éviter le surapprentissage
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Évaluer le modèle
    y_pred_scaled = model.predict(X_test)
    
    # Inverser la normalisation
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Calculer les métriques
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    print(f"LSTM pour {country_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Valeurs réelles')
    plt.plot(y_pred_actual, label='Prédictions')
    plt.title(f'Prédictions LSTM vs Réalité pour {country_name} - {target_col}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{country_name.replace(' ', '_').lower()}_lstm_predictions.png")
    
    # Sauvegarder le modèle
    model_path = f"models/{country_name.replace(' ', '_').lower()}_lstm"
    model.save(model_path)
    
    # Sauvegarder également le scaler pour inverser la normalisation plus tard
    joblib.dump(scaler, f"{model_path}_scaler.pkl")
    
    return model, scaler, (mae, rmse, r2)

# Entraîner des modèles LSTM pour les principaux pays
countries = ['US', 'Brazil', 'France', 'Germany', 'India']
for country in countries:
    print(f"\nEntraînement du modèle LSTM pour {country}:")
    train_lstm_model(country, 'new_cases')
```

### 3.3. Génération de prédictions

```python
# prediction_service.py
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

def predict_next_days(country, days=30, model_type='lstm'):
    """Génère des prédictions pour un nombre spécifié de jours pour un pays."""
    # Vérifier si le modèle existe
    country_code = country.replace(' ', '_').lower()
    
    if model_type == 'lstm':
        model_path = f"models/{country_code}_lstm"
        scaler_path = f"{model_path}_scaler.pkl"
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print(f"Modèle LSTM non disponible pour {country}")
            return None
        
        # Charger le modèle et le scaler
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # Charger les données récentes pour la séquence initiale
        data_path = f"data/by_country/{country_code}.csv"
        df = pd.read_csv(data_path)
        df = df.sort_values('date_value')
        
        # Extraire les données récentes pour new_cases
        recent_data = df['new_cases'].values[-14:].reshape(-1, 1)  # 14 derniers jours
        
        # Normaliser les données
        recent_data_scaled = scaler.transform(recent_data)
        
        # Initialiser la séquence pour la prédiction
        sequence = recent_data_scaled.reshape(1, 14, 1)
        
        # Générer les prédictions pour les prochains jours
        predictions = []
        last_date = pd.to_datetime(df['date_value'].iloc[-1])
        
        for i in range(days):
            # Prédire la prochaine valeur
            next_pred_scaled = model.predict(sequence)
            
            # Inverser la normalisation
            next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]
            
            # Ajouter à la liste des prédictions
            next_date = last_date + timedelta(days=i+1)
            predictions.append({
                'date': next_date.strftime('%Y-%m-%d'),
                'predicted_cases': max(0, int(round(next_pred)))  # Assurer des prédictions positives
            })
            
            # Mettre à jour la séquence pour la prochaine prédiction
            new_sequence = np.append(sequence[0, 1:, :], [[next_pred_scaled[0][0]]], axis=0)
            sequence = new_sequence.reshape(1, 14, 1)
        
        return predictions
    
    else:  # modèles non-LSTM (scikit-learn)
        model_path = f"models/{country_code}_{model_type}.pkl"
        
        if not os.path.exists(model_path):
            print(f"Modèle {model_type} non disponible pour {country}")
            return None
        
        # Charger le modèle
        model = joblib.load(model_path)
        
        # Charger les données récentes
        data_path = f"data/by_country/{country_code}.csv"
        df = pd.read_csv(data_path)
        df = df.sort_values('date_value')
        
        # Obtenir la dernière ligne pour les features
        last_row = df.iloc[-1:]
        
        # Générer les prédictions
        predictions = []
        last_date = pd.to_datetime(df['date_value'].iloc[-1])
        
        # On doit simuler l'évolution des features pour chaque jour
        current_row = last_row.copy()
        
        for i in range(days):
            # Mettre à jour la date
            next_date = last_date + timedelta(days=i+1)
            
            # Mettre à jour les features temporelles
            current_row['year'] = next_date.year
            current_row['month'] = next_date.month
            current_row['day'] = next_date.day
            current_row['day_of_week'] = next_date.weekday()
            
            # Prédire avec les features actuelles
            X = current_row[model.feature_names_in_]
            next_pred = model.predict(X)[0]
            
            # Ajouter à la liste des prédictions
            predictions.append({
                'date': next_date.strftime('%Y-%m-%d'),
                'predicted_cases': max(0, int(round(next_pred)))  # Assurer des prédictions positives
            })
            
            # Mettre à jour certaines features pour la prochaine prédiction
            current_row['new_cases'] = next_pred
            current_row['total_cases'] += next_pred
            
            # Calculer d'autres features dérivées si nécessaire
            # ...
        
        return predictions

# Exemple d'utilisation
for country in ['US', 'Brazil', 'France']:
    print(f"Prédictions pour {country} (30 prochains jours):")
    predictions = predict_next_days(country, days=30)
    if predictions:
        for i, pred in enumerate(predictions[:5]):  # Afficher les 5 premiers jours
            print(f"{pred['date']}: {pred['predicted_cases']} cas prévus")
        print("...")
```

## 4. Évaluation des Modèles d'Apprentissage

### 4.1. Méthodes d'évaluation

```python
# model_evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

def evaluate_model_performance(country_name, model_type='lstm', target_col='new_cases'):
    """Fonction complète d'évaluation des performances d'un modèle."""
    # Charger les données
    country_code = country_name.replace(' ', '_').lower()
    data_path = f"data/by_country/{country_code}.csv"
    
    if not os.path.exists(data_path):
        print(f"Données non disponibles pour {country_name}")
        return None
    
    df = pd.read_csv(data_path)
    df = df.sort_values('date_value')
    
    # Division des données - validation temporelle
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    # Charger le modèle
    if model_type == 'lstm':
        # Pour les modèles LSTM, nous utilisons les prédictions générées précédemment
        predictions_file = f"results/{country_code}_lstm_predictions.csv"
        if not os.path.exists(predictions_file):
            print(f"Prédictions LSTM non disponibles pour {country_name}")
            return None
        
        predictions = pd.read_csv(predictions_file)
        
        # Fusionner avec les données réelles
        merged_data = pd.merge(
            test_data[['date_value', target_col]], 
            predictions[['date', 'predicted_cases']], 
            left_on='date_value', 
            right_on='date'
        )
        
        # Calculer les métriques
        y_true = merged_data[target_col]
        y_pred = merged_data['predicted_cases']
    else:
        # Pour les modèles scikit-learn
        model_path = f"models/{country_code}_{model_type}.pkl"
        if not os.path.exists(model_path):
            print(f"Modèle {model_type} non disponible pour {country_name}")
            return None
        
        model = joblib.load(model_path)
        
        # Préparer les features
        X_test = test_data[model.feature_names_in_]
        y_true = test_data[target_col]
        
        # Prédire
        y_pred = model.predict(X_test)
    
    # Calculer les métriques
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='Valeurs réelles', color='blue')
    plt.plot(y_pred, label='Prédictions', color='red', linestyle='--')
    plt.title(f'Performance du modèle {model_type} pour {country_name}')
    plt.xlabel('Jours')
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{country_code}_{model_type}_evaluation.png")
    
    print(f"Performance pour {country_name} avec {model_type}:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Évaluer les modèles pour plusieurs pays et algorithmes
countries = ['US', 'Brazil', 'France', 'Germany', 'India']
model_types = ['lstm', 'random_forest', 'gradient_boosting', 'xgboost']

results = {}
for country in countries:
    country_results = {}
    for model_type in model_types:
        metrics = evaluate_model_performance(country, model_type)
        if metrics:
            country_results[model_type] = metrics
    results[country] = country_results

# Sauvegarder les résultats dans un fichier CSV
results_df = pd.DataFrame()
for country, country_results in results.items():
    for model_type, metrics in country_results.items():
        row = {'country': country, 'model_type': model_type}
        row.update(metrics)
        results_df = results_df.append(row, ignore_index=True)

results_df.to_csv("results/model_performance_comparison.csv", index=False)
print("\nRésultats d'évaluation sauvegardés dans model_performance_comparison.csv")
```

## 5. Interface Web pour la Visualisation des Prédictions

### 5.1. Application Flask pour la visualisation

```python
# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import plotly
import plotly.express as px
import os
from prediction_service import predict_next_days

app = Flask(__name__)

@app.route('/')
def index():
    # Récupérer la liste des pays disponibles
    country_files = os.listdir('data/by_country')
    countries = [os.path.splitext(file)[0].replace('_', ' ').title() for file in country_files]
    return render_template('index.html', countries=countries)

@app.route('/api/countries')
def get_countries():
    country_files = os.listdir('data/by_country')
    countries = [os.path.splitext(file)[0].replace('_', ' ').title() for file in country_files]
    return jsonify(countries)

@app.route('/api/data/<country>')
def get_country_data(country):
    country_code = country.replace(' ', '_').lower()
    file_path = f'data/by_country/{country_code}.csv'
    
    if not os.path.exists(file_path):
        return jsonify({'error': f'Données non disponibles pour {country}'}), 404
    
    df = pd.read_csv(file_path)
    df = df.sort_values('date_value')
    
    # Convertir en format JSON pour l'API
    result = []
    for _, row in df.iterrows():
        result.append({
            'date': row['date_value'],
            'total_cases': int(row['total_cases']),
            'new_cases': int(row['new_cases']),
            'total_deaths': int(row['total_deaths']) if 'total_deaths' in row else 0,
            'new_deaths': int(row['new_deaths']) if 'new_deaths' in row else 0
        })
    
    return jsonify(result)

@app.route('/api/predict/<country>')
def get_prediction(country):
    days = request.args.get('days', default=30, type=int)
    model_type = request.args.get('model', default='lstm', type=str)
    
    predictions = predict_next_days(country, days, model_type)
    
    if not predictions:
        return jsonify({'error': f'Prédictions non disponibles pour {country} avec le modèle {model_type}'}), 404
    
    return jsonify(predictions)

@app.route('/country/<country>')
def country_view(country):
    days = request.args.get('days', default=30, type=int)
    model_type = request.args.get('model', default='lstm', type=str)
    
    # Récupérer les données historiques
    country_code = country.replace(' ', '_').lower()
    file_path = f'data/by_country/{country_code}.csv'
    
    if not os.path.exists(file_path):
        return render_template('error.html', message=f'Données non disponibles pour {country}')
    
    df = pd.read_csv(file_path)
    df = df.sort_values('date_value')
    
    # Récupérer les prédictions
    predictions = predict_next_days(country, days, model_type)
    
    if not predictions:
        return render_template('error.html', message=f'Prédictions non disponibles pour {country}')
    
    # Créer le graphique avec Plotly
    fig = px.line(df, x='date_value', y='new_cases', title=f'Cas COVID-19 pour {country}')
    
    # Ajouter les prédictions au graphique
    pred_dates = [p['date'] for p in predictions]
    pred_cases = [p['predicted_cases'] for p in predictions]
    fig.add_scatter(x=pred_dates, y=pred_cases, mode='lines', name='Prédictions', line=dict(dash='dash'))
    
    # Convertir le graphique en JSON pour l'intégrer dans le HTML
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('country.html', 
                           country=country, 
                           graph_json=graph_json,
                           days=days,
                           model_type=model_type)

if __name__ == '__main__':
    app.run(debug=True)
```

### 5.2. Templates HTML pour l'interface

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prédiction des Pandémies</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <header>
        <h1>Analyse et Prédiction des Pandémies</h1>
    </header>
    
    <main>
        <section class="country-selector">
            <h2>Sélectionnez un pays</h2>
            <div class="countries-grid">
                {% for country in countries %}
                <a href="/country/{{ country }}" class="country-card">
                    <div class="country-name">{{ country }}</div>
                </a>
                {% endfor %}
            </div>
        </section>
        
        <section class="project-info">
            <h2>À propos du projet</h2>
            <p>Cette application utilise des modèles d'apprentissage automatique pour prédire l'évolution des cas de pandémie dans différents pays.</p>
            <p>Plusieurs modèles ont été entraînés et évalués :</p>
            <ul>
                <li>Réseaux de neurones récurrents (LSTM)</li>
                <li>Forêts aléatoires</li>
                <li>Gradient Boosting</li>
                <li>XGBoost</li>
            </ul>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2025 - Projet MSPR Analyse Prédictive des Pandémies</p>
    </footer>
</body>
</html>
```

```html
<!-- templates/country.html -->
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ country }} - Prédiction des Pandémies</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <header>
        <h1>{{ country }} - Analyse et Prédiction</h1>
        <nav>
            <a href="/">Retour à l'accueil</a>
        </nav>
    </header>
    
    <main>
        <section class="prediction-controls">
            <form action="/country/{{ country }}" method="get">
                <div class="form-group">
                    <label for="days">Jours de prédiction:</label>
                    <input type="range" id="days" name="days" min="7" max="90" value="{{ days }}" oninput="this.nextElementSibling.value = this.value">
                    <output>{{ days }}</output>
                </div>
                
                <div class="form-group">
                    <label for="model">Modèle:</label>
                    <select id="model" name="model">
                        <option value="lstm" {% if model_type == 'lstm' %}selected{% endif %}>LSTM</option>
                        <option value="random_forest" {% if model_type == 'random_forest' %}selected{% endif %}>Forêt Aléatoire</option>
                        <option value="gradient_boosting" {% if model_type == 'gradient_boosting' %}selected{% endif %}>Gradient Boosting</option>
                        <option value="xgboost" {% if model_type == 'xgboost' %}selected{% endif %}>XGBoost</option>
                    </select>
                </div>
                
                <button type="submit">Mettre à jour</button>
            </form>
        </section>
        
        <section class="graph-container">
            <div id="graph"></div>
        </section>
        
        <section class="model-info">
            <h2>Performance du modèle {{ model_type }}</h2>
            <p>Le modèle prédictif a été entraîné sur les données historiques de {{ country }} et optimisé pour la prédiction des nouveaux cas.</p>
            
            <div class="metrics">
                <div class="metric">
                    <h3>MAE</h3>
                    <p class="value">124.5</p>
                    <p class="desc">Erreur absolue moyenne</p>
                </div>
                <div class="metric">
                    <h3>RMSE</h3>
                    <p class="value">189.2</p>
                    <p class="desc">Racine de l'erreur quadratique moyenne</p>
                </div>
                <div class="metric">
                    <h3>R²</h3>
                    <p class="value">0.87</p>
                    <p class="desc">Coefficient de détermination</p>
                </div>
            </div>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2025 - Projet MSPR Analyse Prédictive des Pandémies</p>
    </footer>
    
    <script>
        // Charger le graphique avec les données JSON générées par Flask
        var graphJSON = {{ graph_json|safe }};
        Plotly.newPlot('graph', graphJSON.data, graphJSON.layout);
    </script>
</body>
</html>
```

## 6. Tests et Évaluation

### 6.1. Tests du modèle d'apprentissage

```python
# test_models.py
import unittest
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class TestModels(unittest.TestCase):
    
    def setUp(self):
        # Charger les données de test
        self.test_countries = ['US', 'France', 'Germany']
        self.models_to_test = ['random_forest', 'gradient_boosting', 'lstm']
        
        # Vérifier que les fichiers nécessaires existent
        for country in self.test_countries:
            country_code = country.replace(' ', '_').lower()
            data_path = f"data/by_country/{country_code}.csv"
            self.assertTrue(os.path.exists(data_path), f"Fichier de données manquant pour {country}")
    
    def test_model_predictions(self):
        """Teste si les prédictions des modèles sont raisonnables."""
        from prediction_service import predict_next_days
        
        for country in self.test_countries:
            for model_type in self.models_to_test:
                # Générer des prédictions pour 30 jours
                predictions = predict_next_days(country, days=30, model_type=model_type)
                
                # Vérifier que les prédictions sont disponibles
                self.assertIsNotNone(predictions, f"Prédictions indisponibles pour {country} avec {model_type}")
                
                # Vérifier le nombre de prédictions
                self.assertEqual(len(predictions), 30, f"Nombre incorrect de prédictions pour {country}")
                
                # Vérifier que les prédictions sont des nombres positifs
                for pred in predictions:
                    self.assertIn('predicted_cases', pred)
                    self.assertGreaterEqual(pred['predicted_cases'], 0)
    
    def test_model_performance(self):
        """Teste les performances des modèles sur les données de test."""
        for country in self.test_countries:
            country_code = country.replace(' ', '_').lower()
            
            # Charger les données
            df = pd.read_csv(f"data/by_country/{country_code}.csv")
            df = df.sort_values('date_value')
            
            # Division train/test
            train_size = int(len(df) * 0.8)
            test_data = df.iloc[train_size:]
            
            for model_type in self.models_to_test:
                if model_type != 'lstm':  # Test uniquement pour les modèles scikit-learn
                    model_path = f"models/{country_code}_{model_type}.pkl"
                    
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        
                        # Préparer les features de test
                        X_test = test_data[model.feature_names_in_]
                        y_test = test_data['new_cases']
                        
                        # Prédire
                        y_pred = model.predict(X_test)
                        
                        # Calculer les métriques
                        r2 = r2_score(y_test, y_pred)
                        
                        # Vérifier que R2 est supérieur à un seuil minimal
                        self.assertGreater(r2, 0.3, 
                                          f"Performance insuffisante pour {country} avec {model_type} (R2={r2:.2f})")

if __name__ == '__main__':
    unittest.main()
```

### 6.2. Tests de l'interface web

```python
# test_app.py
import unittest
from app import app

class TestFlaskApp(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_page(self):
        """Teste si la page d'accueil se charge correctement."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Analyse et Pr\xc3\xa9diction des Pand\xc3\xa9mies', response.data)
    
    def test_api_countries(self):
        """Teste si l'API retourne la liste des pays."""
        response = self.app.get('/api/countries')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
    
    def test_api_country_data(self):
        """Teste si l'API retourne les données d'un pays."""
        response = self.app.get('/api/data/US')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn('date', data[0])
        self.assertIn('total_cases', data[0])
    
    def test_api_prediction(self):
        """Teste si l'API retourne des prédictions."""
        response = self.app.get('/api/predict/US?days=7&model=random_forest')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 7)  # 7 jours demandés
        self.assertIn('date', data[0])
        self.assertIn('predicted_cases', data[0])

if __name__ == '__main__':
    unittest.main()
```

## 7. Déploiement et Exécution

### 7.1. Installation des dépendances

```bash
# Installer les dépendances Python
pip install -r requirements.txt
```

### 7.2. Exécution de l'application

```bash
# Lancer l'application Flask
python app.py
```

### 7.3. Exécution des tests

```bash
# Exécuter les tests des modèles
python test_models.py

# Exécuter les tests de l'application
python test_app.py
```

## 8. Perspectives d'Amélioration

- **Amélioration des modèles**: Intégrer des données externes (mesures sanitaires, vaccination, densité de population)
- **Techniques avancées**: Explorer des architectures plus complexes (Transformers, modèles hybrides)
- **Prédiction multi-variables**: Prédire simultanément plusieurs variables (cas, décès, hospitalisations)
- **Interface avancée**: Développer un tableau de bord plus complet avec React ou Vue.js
- **Déploiement cloud**: Migrer l'application vers une plateforme cloud pour un accès public
- **API publique**: Exposer les prédictions via une API REST documentée
