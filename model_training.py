#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'entraînement des modèles de prédiction des pandémies
Ce script correspond à la Phase 2 du projet: Sélection et entraînement des modèles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Créer les dossiers pour sauvegarder les modèles et résultats
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def load_country_data(country):
    """
    Charge les données prétraitées pour un pays spécifique
    """
    country_code = country.replace(' ', '_').lower()
    file_path = f"data/by_country/{country_code}.csv"
    
    if not os.path.exists(file_path):
        print(f"Données non disponibles pour {country}")
        return None
    
    print(f"Chargement des données pour {country}...")
    df = pd.read_csv(file_path)
    
    # Convertir la date en datetime si nécessaire
    if 'date_value' in df.columns and df['date_value'].dtype != 'datetime64[ns]':
        df['date_value'] = pd.to_datetime(df['date_value'])
        
    # Trier par date
    if 'date_value' in df.columns:
        df = df.sort_values('date_value')
    
    print(f"Chargé {len(df)} entrées pour {country}")
    return df

def prepare_features(df, target_col='new_cases'):
    """
    Prépare les features et la cible pour l'entraînement du modèle
    """
    print(f"Préparation des features avec la cible: {target_col}")
    
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
    
    print(f"Features sélectionnées ({len(available_features)}): {available_features}")
    
    # Créer le dataframe de features
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Gérer les valeurs manquantes
    X = X.fillna(0)
    
    # Gérer les valeurs infinies ou trop grandes
    for col in X.columns:
        # Remplacer les valeurs infinies par des 0
        X[col] = X[col].replace([np.inf, -np.inf], 0)
        
        # Détecter et remplacer les valeurs extrêmes
        # On considère comme extrême toute valeur > 1e9 ou < -1e9
        X[col] = X[col].apply(lambda x: 0 if abs(x) > 1e9 else x)
    
    print("Nettoyage des valeurs infinies et extrêmes effectué")
    
    return X, y

def train_test_split_ts(X, y, test_size=0.2):
    """
    Division des données en ensembles d'entraînement et de test
    adaptée aux séries temporelles (pas de mélange aléatoire)
    """
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"Division en {len(X_train)} échantillons d'entraînement et {len(X_test)} échantillons de test")
    return X_train, X_test, y_train, y_test

def train_classical_models(X_train, X_test, y_train, y_test, country):
    """
    Entraîne et évalue plusieurs modèles classiques de ML
    """
    print(f"\nEntraînement des modèles classiques pour {country}...")
    
    # Définir les modèles à évaluer
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Évaluer chaque modèle
    results = {}
    for name, model in models.items():
        print(f"Entraînement du modèle: {name}")
        
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédiction
        y_pred = model.predict(X_test)
        
        # Métriques d'évaluation
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Enregistrer les résultats
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        # Sauvegarder le modèle
        country_code = country.replace(' ', '_').lower()
        model_path = f"models/{country_code}_{name}.pkl"
        joblib.dump(model, model_path)
        print(f"  Modèle sauvegardé: {model_path}")
        
        # Visualiser les prédictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Réel', color='blue')
        plt.plot(y_pred, label='Prédiction', color='red', linestyle='--')
        plt.title(f'Prédictions {name} pour {country}')
        plt.xlabel('Échantillons de test')
        plt.ylabel('Nouveaux cas')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/{country_code}_{name}_predictions.png')
        plt.close()
    
    # Comparer les performances des modèles
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), [results[model]['RMSE'] for model in results])
    plt.title(f'Comparaison des RMSE par modèle pour {country}')
    plt.ylabel('RMSE (erreur)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/{country_code}_models_comparison.png')
    plt.close()
    
    return results

def create_lstm_sequences(data, seq_length=14):
    """
    Crée des séquences pour l'entraînement du modèle LSTM
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_lstm_model(df, country, target_col='new_cases', seq_length=14):
    """
    Entraîne un modèle LSTM pour prédire les données temporelles
    """
    print(f"\nEntraînement du modèle LSTM pour {country}...")
    
    # Préparer les données pour LSTM - ne nécessite que la série temporelle cible
    data = df[target_col].values.reshape(-1, 1)
    
    # Normaliser les données pour LSTM
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Créer des séquences
    X, y = create_lstm_sequences(data_scaled, seq_length)
    
    # Reshape pour le format attendu par LSTM [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Diviser en ensembles d'entraînement et de test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Préparé {len(X_train)} séquences d'entraînement et {len(X_test)} séquences de test")
    
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
    
    # Inverser la normalisation pour comparer avec les valeurs réelles
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Calculer les métriques
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    print(f"LSTM pour {country} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Réel', color='blue')
    plt.plot(y_pred_actual, label='Prédiction', color='red', linestyle='--')
    plt.title(f'Prédictions LSTM pour {country}')
    plt.xlabel('Échantillons de test')
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    
    country_code = country.replace(' ', '_').lower()
    plt.savefig(f'results/{country_code}_lstm_predictions.png')
    plt.close()
    
    # Visualiser l'historique d'entraînement
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Perte (entraînement)')
    plt.plot(history.history['val_loss'], label='Perte (validation)')
    plt.title(f'Historique d\'entraînement LSTM pour {country}')
    plt.xlabel('Epoch')
    plt.ylabel('Perte (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{country_code}_lstm_training_history.png')
    plt.close()
    
    # Sauvegarder le modèle
    model_path = f"models/{country_code}_lstm.keras"
    model.save(model_path)
    
    # Sauvegarder le scaler pour pouvoir inverser la normalisation plus tard
    joblib.dump(scaler, f"models/{country_code}_lstm_scaler.pkl")
    
    print(f"Modèle LSTM et scaler sauvegardés dans {model_path}")
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

def optimize_hyperparameters(X_train, y_train, country, model_type='random_forest'):
    """
    Optimise les hyperparamètres pour un modèle spécifique
    """
    print(f"\nOptimisation des hyperparamètres pour {model_type} ({country})...")
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    else:
        print(f"Type de modèle non pris en charge pour l'optimisation: {model_type}")
        return None
    
    # Utiliser une validation croisée adaptée aux séries temporelles
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Créer et exécuter la recherche par grille
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Afficher les meilleurs paramètres
    print(f"Meilleurs paramètres trouvés: {grid_search.best_params_}")
    print(f"Meilleur score: {-grid_search.best_score_:.2f} RMSE")
    
    # Sauvegarder le meilleur modèle
    best_model = grid_search.best_estimator_
    country_code = country.replace(' ', '_').lower()
    model_path = f"models/{country_code}_{model_type}_optimized.pkl"
    joblib.dump(best_model, model_path)
    print(f"Meilleur modèle sauvegardé: {model_path}")
    
    return best_model, grid_search.best_params_

def main():
    """
    Fonction principale exécutant le pipeline d'entraînement des modèles
    """
    print("=== DÉBUT DU PROCESSUS D'ENTRAÎNEMENT DES MODÈLES ===")
    print(f"Date et heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Liste des pays principaux à traiter
    # Utiliser les noms exacts comme dans les fichiers
    main_countries = ['afghanistan', 'albania', 'algeria', 'andorra', 'angola']
    
    # Convertir les noms de fichier en noms de pays présentables
    main_countries = [country.replace('_', ' ').title() for country in main_countries]
    
    # Stocker les résultats pour tous les pays et modèles
    all_results = {}
    
    for country in main_countries:
        print(f"\n=== Traitement pour {country} ===")
        
        # Étape 1: Charger les données du pays
        df = load_country_data(country)
        if df is None:
            continue
        
        # Étape 2: Préparer les features et la cible
        X, y = prepare_features(df, target_col='new_cases')
        
        # Étape 3: Diviser en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split_ts(X, y, test_size=0.2)
        
        # Étape 4: Entraîner les modèles classiques
        classical_results = train_classical_models(X_train, X_test, y_train, y_test, country)
        
        # Étape 5: Entraîner un modèle LSTM
        lstm_results = train_lstm_model(df, country, target_col='new_cases')
        
        # Initialiser le dictionnaire des résultats pour ce pays
        country_results = {
            **classical_results,
            'lstm': lstm_results
        }
        
        try:
            # Étape 6: Optimiser les hyperparamètres du meilleur modèle classique
            # Trouver le meilleur modèle classique basé sur RMSE
            best_model_name = min(classical_results, key=lambda x: classical_results[x]['RMSE'])
            print(f"\nMeilleur modèle classique pour {country}: {best_model_name}")
            
            # Optimiser ce modèle seulement s'il est pris en charge
            optimization_result = optimize_hyperparameters(X_train, y_train, country, model_type=best_model_name)
            
            # Vérifier si l'optimisation a réussi
            if optimization_result is not None:
                optimized_model, best_params = optimization_result
                
                # Étape 7: Évaluer le modèle optimisé
                y_pred_optimized = optimized_model.predict(X_test)
                optimized_results = {
                    'MAE': mean_absolute_error(y_test, y_pred_optimized),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_optimized)),
                    'R2': r2_score(y_test, y_pred_optimized)
                }
                
                print(f"Modèle optimisé {best_model_name} - MAE: {optimized_results['MAE']:.2f}, "
                      f"RMSE: {optimized_results['RMSE']:.2f}, R²: {optimized_results['R2']:.4f}")
                
                # Étape 8: Visualiser les résultats du modèle optimisé
                plt.figure(figsize=(12, 6))
                plt.plot(y_test.values, label='Réel', color='blue')
                plt.plot(y_pred_optimized, label='Prédiction (optimisé)', color='green', linestyle='--')
                plt.title(f'Prédictions {best_model_name} optimisé pour {country}')
                plt.xlabel('Échantillons de test')
                plt.ylabel('Nouveaux cas')
                plt.legend()
                plt.tight_layout()
                country_code = country.replace(' ', '_').lower()
                plt.savefig(f'results/{country_code}_{best_model_name}_optimized_predictions.png')
                plt.close()
                
                # Ajouter les résultats optimisés
                country_results[f"{best_model_name}_optimized"] = optimized_results
            else:
                print(f"Pas d'optimisation possible pour {best_model_name}, utilisation du modèle standard")
        except Exception as e:
            print(f"Erreur lors de l'optimisation des hyperparamètres pour {country}: {str(e)}")
            print("Poursuite du traitement avec les modèles standards uniquement")
        
        # Ajouter les résultats de ce pays à all_results
        all_results[country] = country_results
    
    # Sauvegarder tous les résultats dans un fichier CSV
    results_df = pd.DataFrame()
    for country, models in all_results.items():
        for model_name, metrics in models.items():
            try:
                row = {
                    'country': country,
                    'model': model_name,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2']
                }
                # Utiliser concat au lieu de append qui est déprécié
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
            except Exception as e:
                print(f"Erreur lors de l'ajout des résultats pour {country}, {model_name}: {str(e)}")
    
    results_df.to_csv('results/model_performance_comparison.csv', index=False)
    print("\nRésultats d'évaluation sauvegardés dans results/model_performance_comparison.csv")
    
    print("\n=== FIN DU PROCESSUS D'ENTRAÎNEMENT DES MODÈLES ===")
    print(f"Date et heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nLes modèles sont prêts pour la phase d'évaluation et de déploiement.")

if __name__ == "__main__":
    main()
