#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'évaluation des modèles de prédiction des pandémies
Ce script correspond à la Phase 3 du projet: Évaluation et analyse des modèles
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Créer le dossier pour les résultats d'évaluation
os.makedirs('evaluation', exist_ok=True)

def load_country_data(country):
    """
    Charge les données pour un pays spécifique
    """
    country_code = country.replace(' ', '_').lower()
    file_path = f'data/by_country/{country_code}.csv'
    
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
    Prépare les features et la cible pour l'évaluation du modèle
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

def load_models(country):
    """
    Charge les modèles entraînés pour un pays spécifique
    """
    country_code = country.replace(' ', '_').lower()
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        print(f"Le dossier {models_dir} n'existe pas")
        return None
    
    models = {}
    model_files = [f for f in os.listdir(models_dir) if f.startswith(country_code) and f.endswith('.pkl')]
    
    print(f"Modèles disponibles pour {country}: {len(model_files)}")
    
    for model_file in model_files:
        model_name = model_file.replace(f"{country_code}_", "").replace(".pkl", "")
        try:
            model = joblib.load(os.path.join(models_dir, model_file))
            models[model_name] = model
            print(f"  Modèle chargé: {model_name}")
        except Exception as e:
            print(f"  Erreur lors du chargement du modèle {model_file}: {str(e)}")
    
    # Vérifier s'il y a un modèle LSTM
    lstm_path = os.path.join(models_dir, f"{country_code}_lstm.keras")
    if os.path.exists(lstm_path):
        try:
            lstm_model = keras.models.load_model(lstm_path)
            models['lstm'] = lstm_model
            
            # Charger aussi le scaler pour LSTM
            scaler_path = os.path.join(models_dir, f"{country_code}_lstm_scaler.pkl")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                models['lstm_scaler'] = scaler
                print(f"  Modèle LSTM et scaler chargés")
            else:
                print(f"  Scaler LSTM introuvable: {scaler_path}")
        except Exception as e:
            print(f"  Erreur lors du chargement du modèle LSTM: {str(e)}")
    
    return models

def create_lstm_sequences(data, seq_length=14):
    """
    Crée des séquences pour l'évaluation du modèle LSTM
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def evaluate_models(country, test_size=0.2, target_col='new_cases'):
    """
    Évalue tous les modèles disponibles pour un pays
    """
    print(f"=== Évaluation des modèles pour {country} ===")
    
    # Charger les données
    df = load_country_data(country)
    if df is None:
        return None
    
    # Préparer les features
    X, y = prepare_features(df, target_col)
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split_ts(X, y, test_size)
    
    # Charger les modèles
    models = load_models(country)
    if not models:
        print(f"Aucun modèle disponible pour {country}")
        return None
    
    # Évaluer chaque modèle
    results = {}
    
    for model_name, model in models.items():
        if model_name == 'lstm_scaler':
            continue  # Ignorer le scaler, ce n'est pas un modèle
        
        print(f"\nÉvaluation du modèle: {model_name}")
        
        try:
            if model_name == 'lstm':
                # Préparation spéciale pour LSTM
                data = y.values.reshape(-1, 1)
                scaler = models.get('lstm_scaler')
                
                if scaler is None:
                    print("  Scaler manquant pour le modèle LSTM, impossible d'évaluer")
                    continue
                
                data_scaled = scaler.transform(data)
                
                # Créer des séquences
                seq_length = 14  # Même valeur que pour l'entraînement
                X_seq, y_seq = create_lstm_sequences(data_scaled, seq_length)
                
                # Séparer en train/test
                train_size = int(len(X_seq) * (1 - test_size))
                X_test_seq = X_seq[train_size:]
                y_test_seq = y_seq[train_size:]
                
                # Reshape pour LSTM [samples, timesteps, features]
                X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1], 1)
                
                # Prédire
                y_pred_scaled = model.predict(X_test_seq)
                
                # Inverser la normalisation
                y_test_actual = scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
                y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
            else:
                # Modèles classiques
                y_pred = model.predict(X_test)
                y_test_actual = y_test.values
            
            # Calculer les métriques
            mae = mean_absolute_error(y_test_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
            r2 = r2_score(y_test_actual, y_pred)
            
            results[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'predictions': y_pred,
                'actual': y_test_actual
            }
            
            print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            # Créer des visualisations comparatives
            plt.figure(figsize=(12, 6))
            plt.plot(y_test_actual, label='Réel', color='blue')
            plt.plot(y_pred, label='Prédiction', color='red', linestyle='--')
            plt.title(f'Prédictions vs Réalité - {model_name.capitalize()} pour {country}')
            plt.xlabel('Échantillons de test')
            plt.ylabel(target_col)
            plt.legend()
            plt.tight_layout()
            
            # Sauvegarder dans le dossier d'évaluation
            country_code = country.replace(' ', '_').lower()
            plt.savefig(f'evaluation/{country_code}_{model_name}_evaluation.png')
            plt.close()
            
        except Exception as e:
            print(f"  Erreur lors de l'évaluation du modèle {model_name}: {str(e)}")
    
    # Comparer les performances des modèles
    if results:
        print("\nComparaison des performances des modèles:")
        
        # Créer un DataFrame pour faciliter la comparaison
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'MAE': [results[model]['MAE'] for model in results],
            'RMSE': [results[model]['RMSE'] for model in results],
            'R2': [results[model]['R2'] for model in results]
        })
        
        # Trier par RMSE (croissant)
        comparison_df = comparison_df.sort_values('RMSE')
        
        print(comparison_df)
        
        # Sauvegarder les résultats dans un CSV
        country_code = country.replace(' ', '_').lower()
        comparison_df.to_csv(f'evaluation/{country_code}_model_comparison.csv', index=False)
        
        # Créer des visualisations comparatives
        plt.figure(figsize=(12, 6))
        plt.bar(comparison_df['Model'], comparison_df['RMSE'], color='orange')
        plt.title(f'Comparaison des RMSE par modèle pour {country}')
        plt.xlabel('Modèle')
        plt.ylabel('RMSE (erreur)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'evaluation/{country_code}_rmse_comparison.png')
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.bar(comparison_df['Model'], comparison_df['R2'], color='green')
        plt.title(f'Comparaison des R² par modèle pour {country}')
        plt.xlabel('Modèle')
        plt.ylabel('R² (coefficient de détermination)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'evaluation/{country_code}_r2_comparison.png')
        plt.close()
        
        # Identifier le meilleur modèle
        best_model = comparison_df.iloc[0]['Model']
        print(f"\nMeilleur modèle pour {country}: {best_model} avec RMSE = {comparison_df.iloc[0]['RMSE']:.2f}")
        
        return results, comparison_df
    
    return None, None

def feature_importance_analysis(country):
    """
    Analyse l'importance des features pour les modèles qui le supportent
    """
    print(f"\n=== Analyse de l'importance des features pour {country} ===")
    
    # Charger les données
    df = load_country_data(country)
    if df is None:
        return
    
    # Préparer les features
    X, y = prepare_features(df, 'new_cases')
    
    # Charger les modèles
    models = load_models(country)
    if not models:
        print(f"Aucun modèle disponible pour {country}")
        return
    
    # Analyser l'importance des features pour les modèles qui le supportent
    for model_name, model in models.items():
        if model_name in ['random_forest', 'gradient_boosting', 'xgboost'] and hasattr(model, 'feature_importances_'):
            print(f"\nImportance des features pour {model_name}:")
            
            # Créer un DataFrame avec les importances
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Afficher les 10 features les plus importantes
            print(feature_importance.head(10))
            
            # Sauvegarder dans un CSV
            country_code = country.replace(' ', '_').lower()
            feature_importance.to_csv(f'evaluation/{country_code}_{model_name}_feature_importance.csv', index=False)
            
            # Visualiser
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
            plt.title(f'Importance des features - {model_name.capitalize()} pour {country}')
            plt.tight_layout()
            plt.savefig(f'evaluation/{country_code}_{model_name}_feature_importance.png')
            plt.close()

def generate_forecast(country, days=30, model_type=None):
    """
    Génère des prévisions futures pour un pays donné
    """
    print(f"\n=== Génération de prévisions pour {country} ({days} jours) ===")
    
    # Charger les données
    df = load_country_data(country)
    if df is None:
        return None
    
    # Préparer les features
    X, y = prepare_features(df, 'new_cases')
    
    # Charger les modèles
    models = load_models(country)
    if not models:
        print(f"Aucun modèle disponible pour {country}")
        return None
    
    # Si aucun modèle spécifique n'est fourni, utiliser le meilleur modèle
    if model_type is None or model_type not in models:
        # Évaluer tous les modèles pour trouver le meilleur
        _, comparison_df = evaluate_models(country)
        if comparison_df is not None and not comparison_df.empty:
            model_type = comparison_df.iloc[0]['Model']
            print(f"Utilisation du meilleur modèle: {model_type}")
        else:
            # Utiliser un modèle par défaut
            model_types = list(models.keys())
            if 'gradient_boosting' in model_types:
                model_type = 'gradient_boosting'
            elif 'random_forest' in model_types:
                model_type = 'random_forest'
            elif len(model_types) > 0 and model_types[0] != 'lstm_scaler':
                model_type = model_types[0]
            else:
                print("Aucun modèle approprié disponible")
                return None
    
    model = models.get(model_type)
    if model is None:
        print(f"Modèle {model_type} non disponible")
        return None
    
    # Obtenir la dernière date
    if 'date_value' in df.columns:
        last_date = pd.to_datetime(df['date_value'].max())
        print(f"Dernière date disponible: {last_date.strftime('%Y-%m-%d')}")
        
        # Créer un dataframe avec les dates futures
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        forecast_df = pd.DataFrame({'date_value': future_dates})
        
        # Ajouter les composantes de date
        forecast_df['year'] = forecast_df['date_value'].dt.year
        forecast_df['month'] = forecast_df['date_value'].dt.month
        forecast_df['day'] = forecast_df['date_value'].dt.day
        forecast_df['day_of_week'] = forecast_df['date_value'].dt.dayofweek
        forecast_df['is_weekend'] = forecast_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        forecast_df['quarter'] = forecast_df['date_value'].dt.quarter
    
    # Générer des prévisions
    if model_type == 'lstm':
        # Préparation spéciale pour LSTM
        scaler = models.get('lstm_scaler')
        if scaler is None:
            print("Scaler manquant pour le modèle LSTM, impossible de générer des prévisions")
            return None
        
        # Préparer les données
        data = y.values.reshape(-1, 1)
        data_scaled = scaler.transform(data)
        
        # Utiliser les dernières valeurs pour générer des prévisions
        seq_length = 14  # Même valeur que pour l'entraînement
        last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)
        
        # Générer des prévisions jour par jour
        predictions_scaled = []
        current_seq = last_sequence.copy()
        
        for _ in range(days):
            # Prédire la prochaine valeur
            next_val_scaled = model.predict(current_seq)[0]
            predictions_scaled.append(next_val_scaled[0])
            
            # Mettre à jour la séquence en supprimant la première valeur et en ajoutant la prédiction
            current_seq = np.append(current_seq[:, 1:, :], [[next_val_scaled]], axis=1)
        
        # Inverser la normalisation
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    else:
        # Pour les modèles classiques, nous avons besoin de préparer des features similaires
        # aux données d'entraînement pour chaque jour futur
        
        # On commence par extraire les dernières valeurs connues
        last_row = X.iloc[-1].copy()
        
        # Initialiser les prédictions
        predictions = []
        
        for i in range(days):
            # Créer un nouveau dataframe pour cette prédiction
            pred_X = pd.DataFrame([last_row.copy()])
            
            # Mettre à jour les features temporelles si disponibles
            if 'date_value' in df.columns:
                future_date = forecast_df.iloc[i]
                if 'year' in pred_X.columns and 'year' in future_date:
                    pred_X['year'] = future_date['year']
                if 'month' in pred_X.columns and 'month' in future_date:
                    pred_X['month'] = future_date['month']
                if 'day' in pred_X.columns and 'day' in future_date:
                    pred_X['day'] = future_date['day']
                if 'day_of_week' in pred_X.columns and 'day_of_week' in future_date:
                    pred_X['day_of_week'] = future_date['day_of_week']
                if 'is_weekend' in pred_X.columns and 'is_weekend' in future_date:
                    pred_X['is_weekend'] = future_date['is_weekend']
                if 'quarter' in pred_X.columns and 'quarter' in future_date:
                    pred_X['quarter'] = future_date['quarter']
            
            # Gérer les valeurs infinies ou manquantes
            pred_X = pred_X.fillna(0)
            for col in pred_X.columns:
                pred_X[col] = pred_X[col].replace([np.inf, -np.inf], 0)
            
            # Prédire
            try:
                pred = model.predict(pred_X)[0]
                predictions.append(pred)
                
                # Mettre à jour last_row pour la prochaine prédiction
                # (simule l'évolution des variables au fil du temps)
                if 'new_cases' in last_row:
                    last_row['new_cases'] = pred
                if 'total_cases' in last_row:
                    last_row['total_cases'] += pred
            except Exception as e:
                print(f"Erreur lors de la prédiction du jour {i+1}: {str(e)}")
                # En cas d'erreur, utiliser la dernière prédiction ou 0
                predictions.append(predictions[-1] if predictions else 0)
    
    # Créer un dataframe avec les prédictions
    if 'date_value' in df.columns:
        forecast_df['predicted_new_cases'] = predictions
    else:
        forecast_df = pd.DataFrame({
            'day': range(1, days + 1),
            'predicted_new_cases': predictions
        })
    
    # Visualiser les prédictions
    plt.figure(figsize=(14, 7))
    
    # Afficher les données historiques
    historical_days = 60  # Afficher les 60 derniers jours de données historiques
    if 'date_value' in df.columns:
        historical_data = df[['date_value', 'new_cases']].tail(historical_days)
        plt.plot(historical_data['date_value'], historical_data['new_cases'], 
                 label='Données historiques', color='blue')
        plt.plot(forecast_df['date_value'], forecast_df['predicted_new_cases'], 
                 label=f'Prédictions ({model_type})', color='red', linestyle='--')
        plt.axvline(x=last_date, color='green', linestyle='-', alpha=0.5, 
                    label='Limite données historiques/prédictions')
    else:
        plt.plot(range(1, historical_days + 1), y.tail(historical_days), 
                 label='Données historiques', color='blue')
        plt.plot(range(historical_days + 1, historical_days + days + 1), predictions, 
                 label=f'Prédictions ({model_type})', color='red', linestyle='--')
        plt.axvline(x=historical_days, color='green', linestyle='-', alpha=0.5, 
                    label='Limite données historiques/prédictions')
    
    plt.title(f'Prévision des nouveaux cas pour {country} - {days} jours')
    plt.xlabel('Date')
    plt.ylabel('Nouveaux cas')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Sauvegarder la visualisation
    country_code = country.replace(' ', '_').lower()
    plt.savefig(f'evaluation/{country_code}_{model_type}_forecast_{days}days.png')
    plt.close()
    
    # Sauvegarder les prédictions dans un CSV
    forecast_df.to_csv(f'evaluation/{country_code}_{model_type}_forecast_{days}days.csv', index=False)
    
    print(f"Prévisions générées et sauvegardées pour {country} ({model_type}, {days} jours)")
    return forecast_df

def main():
    """
    Fonction principale exécutant le pipeline d'évaluation des modèles
    """
    print("=== DÉBUT DU PROCESSUS D'ÉVALUATION DES MODÈLES ===")
    print(f"Date et heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Liste des pays pour lesquels nous avons des modèles
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"Le dossier {models_dir} n'existe pas")
        return
    
    # Extraire les pays à partir des noms de fichiers de modèles
    model_files = os.listdir(models_dir)
    countries = set()
    for file in model_files:
        if file.endswith('.pkl') or file.endswith('.keras'):
            parts = file.split('_')
            if len(parts) > 1:
                country = parts[0]
                countries.add(country)
    
    if not countries:
        print("Aucun modèle trouvé")
        return
    
    countries = sorted(list(countries))
    print(f"Pays avec des modèles disponibles: {countries}")
    
    # Évaluer les modèles pour chaque pays
    all_results = {}
    for country in countries:
        country_name = country.replace('_', ' ').title()
        print(f"\n{'='*40}")
        print(f"ÉVALUATION POUR {country_name}")
        print(f"{'='*40}")
        
        # Évaluer tous les modèles
        results, comparison = evaluate_models(country_name)
        if results:
            all_results[country_name] = {
                'results': results,
                'comparison': comparison
            }
            
            # Analyser l'importance des features
            feature_importance_analysis(country_name)
            
            # Générer des prévisions (30 jours)
            generate_forecast(country_name, days=30)
    
    # Comparaison globale des performances des modèles entre les pays
    if all_results:
        print("\n=== COMPARAISON GLOBALE DES PERFORMANCES ===")
        
        # Créer un DataFrame pour la comparaison
        global_df = pd.DataFrame(columns=['Country', 'Model', 'MAE', 'RMSE', 'R2'])
        
        for country, data in all_results.items():
            comparison = data['comparison']
            if comparison is not None:
                for _, row in comparison.iterrows():
                    global_df = pd.concat([global_df, pd.DataFrame([{
                        'Country': country,
                        'Model': row['Model'],
                        'MAE': row['MAE'],
                        'RMSE': row['RMSE'],
                        'R2': row['R2']
                    }])], ignore_index=True)
        
        # Sauvegarder la comparaison globale
        global_df.to_csv('evaluation/global_model_comparison.csv', index=False)
        
        # Visualisation de la comparaison globale
        plt.figure(figsize=(14, 8))
        pivot_df = global_df.pivot(index='Country', columns='Model', values='R2')
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('Comparaison des R² par modèle et par pays')
        plt.tight_layout()
        plt.savefig('evaluation/global_r2_comparison.png')
        plt.close()
        
        # Trouver le meilleur modèle global
        best_models = global_df.loc[global_df.groupby('Country')['R2'].idxmax()]
        print("Meilleurs modèles par pays:")
        print(best_models[['Country', 'Model', 'R2']])
        
        # Sauvegarder les meilleurs modèles
        best_models.to_csv('evaluation/best_models_by_country.csv', index=False)
    
    print("\n=== FIN DU PROCESSUS D'ÉVALUATION DES MODÈLES ===")
    print(f"Date et heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
