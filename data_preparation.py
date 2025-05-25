#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'exploration et de préparation des données de pandémies
Ce script correspond à la Phase 1 du projet: Préparation des données
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Créer les dossiers de structure si nécessaires
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/by_country', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

def load_data(file_path='full_grouped.csv'):
    """
    Charge les données depuis le fichier CSV et effectue un premier diagnostic
    """
    print(f"Chargement des données depuis {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dimensions du dataset: {df.shape}")
    
    # Informations sur les types de données
    print("\nInformations sur les types de données:")
    print(df.info())
    
    # Statistiques descriptives
    print("\nStatistiques descriptives:")
    print(df.describe())
    
    # Vérifier les valeurs manquantes
    print("\nNombre de valeurs manquantes par colonne:")
    print(df.isnull().sum())
    
    return df

def explore_data(df):
    """
    Exploration des données avec visualisations
    """
    print("\nExploration des données...")
    
    # Convertir la colonne de date en datetime si elle ne l'est pas déjà
    if df['date_value'].dtype != 'datetime64[ns]':
        df['date_value'] = pd.to_datetime(df['date_value'])
    
    # Distribution des pays
    countries_count = df['country'].value_counts()
    print(f"\nNombre de pays uniques: {len(countries_count)}")
    print("\nTop 10 des pays avec le plus de données:")
    print(countries_count.head(10))
    
    # Créer un graphique de la distribution des pays
    plt.figure(figsize=(12, 8))
    top_countries = countries_count.head(20).index
    sns.countplot(y=df['country'][df['country'].isin(top_countries)], order=top_countries)
    plt.title('Top 20 des pays par nombre d\'entrées')
    plt.tight_layout()
    plt.savefig('visualizations/top_countries_distribution.png')
    plt.close()
    
    # Évolution du nombre de cas pour les principaux pays
    plt.figure(figsize=(14, 8))
    top_5_countries = countries_count.head(5).index
    for country in top_5_countries:
        country_data = df[df['country'] == country]
        plt.plot(country_data['date_value'], country_data['total_cases'], label=country)
    
    plt.title('Évolution des cas totaux pour les 5 principaux pays')
    plt.xlabel('Date')
    plt.ylabel('Nombre de cas')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/cases_evolution_top5.png')
    plt.close()
    
    # Distribution des cas par pandémie
    if 'id_pandemic' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='id_pandemic', y='total_cases', data=df)
        plt.title('Distribution des cas par type de pandémie')
        plt.yscale('log')  # Échelle logarithmique pour mieux visualiser
        plt.savefig('visualizations/cases_by_pandemic_type.png')
        plt.close()
    
    # Corrélation entre les variables numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matrice de corrélation des variables numériques')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')
    plt.close()
    
    return df

def handle_missing_values(df):
    """
    Détection et traitement des valeurs manquantes
    """
    print("\nTraitement des valeurs manquantes...")
    
    # Compter les valeurs manquantes avant traitement
    missing_before = df.isnull().sum()
    
    # Stratégies de traitement pour chaque colonne
    if df['date_value'].isnull().sum() > 0:
        print("Suppression des lignes avec dates manquantes (données temporelles cruciales)")
        df = df.dropna(subset=['date_value'])
    
    # Pour les cas/décès, remplacer par 0 ou par la médiane selon le contexte
    for col in ['total_cases', 'new_cases', 'total_deaths', 'new_deaths']:
        if col in df.columns and df[col].isnull().sum() > 0:
            print(f"Remplacement des valeurs manquantes dans {col} par 0")
            df[col] = df[col].fillna(0)
    
    # Compter les valeurs manquantes après traitement
    missing_after = df.isnull().sum()
    
    print("\nRésumé du traitement des valeurs manquantes:")
    print(pd.DataFrame({
        'Avant': missing_before,
        'Après': missing_after,
        'Différence': missing_before - missing_after
    }))
    
    return df

def handle_outliers(df):
    """
    Détection et traitement des valeurs aberrantes
    """
    print("\nTraitement des valeurs aberrantes...")
    
    # Identification des valeurs aberrantes par colonne numérique
    numeric_cols = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    for col in numeric_cols:
        # Vérifier les valeurs négatives qui n'ont pas de sens dans ce contexte
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"Correction de {neg_count} valeurs négatives dans {col}")
            df[col] = df[col].clip(lower=0)
        
        # Détection des outliers par méthode IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Détecté {len(outliers)} outliers dans {col} (limites: {lower_bound:.2f}, {upper_bound:.2f})")
        
        # Visualisation des outliers
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=df[col])
        plt.title(f'Distribution et outliers pour {col}')
        plt.savefig(f'visualizations/outliers_{col}.png')
        plt.close()
        
        # Note: Nous ne supprimons pas automatiquement les outliers car ils peuvent
        # représenter des événements réels importants (pics de pandémie)
    
    return df

def create_derived_features(df):
    """
    Création de features dérivées pour l'enrichissement du modèle
    """
    print("\nCréation de features dérivées...")
    
    # S'assurer que date_value est au format datetime
    if df['date_value'].dtype != 'datetime64[ns]':
        df['date_value'] = pd.to_datetime(df['date_value'])
    
    # Extraire des caractéristiques temporelles
    df['year'] = df['date_value'].dt.year
    df['month'] = df['date_value'].dt.month
    df['day'] = df['date_value'].dt.day
    df['day_of_week'] = df['date_value'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['quarter'] = df['date_value'].dt.quarter
    
    # Calculer le taux de mortalité (si les colonnes nécessaires existent)
    if 'total_deaths' in df.columns and 'total_cases' in df.columns:
        df['mortality_rate'] = np.where(df['total_cases'] > 0, 
                                       df['total_deaths'] / df['total_cases'], 
                                       0)
    
    # Trier les données par pays et date pour calculer des features basées sur l'historique
    df = df.sort_values(['country', 'date_value'])
    
    # Calculer les taux de croissance et moyennes mobiles par pays
    for country in df['country'].unique():
        country_mask = df['country'] == country
        
        # Taux de croissance sur 7 jours
        if 'total_cases' in df.columns:
            df.loc[country_mask, 'cases_growth_rate_7d'] = df.loc[country_mask, 'total_cases'].pct_change(periods=7)
        
        if 'total_deaths' in df.columns:
            df.loc[country_mask, 'deaths_growth_rate_7d'] = df.loc[country_mask, 'total_deaths'].pct_change(periods=7)
        
        # Moyennes mobiles sur 7 jours
        if 'new_cases' in df.columns:
            df.loc[country_mask, 'new_cases_ma7'] = df.loc[country_mask, 'new_cases'].rolling(window=7, min_periods=1).mean()
        
        if 'new_deaths' in df.columns:
            df.loc[country_mask, 'new_deaths_ma7'] = df.loc[country_mask, 'new_deaths'].rolling(window=7, min_periods=1).mean()
    
    # Afficher les nouvelles colonnes ajoutées
    new_features = [col for col in df.columns if col not in ['date_value', 'country', 'total_cases', 'total_deaths', 'new_cases', 'new_deaths', 'id_pandemic']]
    print(f"Nouvelles features créées: {new_features}")
    
    return df

def normalize_data(df):
    """
    Normalisation des données numériques
    """
    print("\nNormalisation des données numériques...")
    
    # Colonnes à normaliser (exclure les identifiants et dates)
    exclude_cols = ['date_value', 'country', 'id_pandemic', 'year', 'month', 'day', 'day_of_week', 'is_weekend', 'quarter']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    normalize_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Colonnes à normaliser: {normalize_cols}")
    
    # Créer une copie pour les données normalisées
    df_normalized = df.copy()
    
    # Normaliser chaque colonne séparément par la méthode Min-Max
    for col in normalize_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        
        # Éviter la division par zéro
        if max_val > min_val:
            df_normalized[col + '_normalized'] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col + '_normalized'] = 0
    
    return df_normalized

def split_data(df):
    """
    Division des données en ensembles d'entraînement/validation/test
    et par pays pour l'entraînement spécifique
    """
    print("\nDivision des données...")
    
    # Division par pays
    os.makedirs('data/by_country', exist_ok=True)
    
    countries = df['country'].unique()
    print(f"Division des données pour {len(countries)} pays...")
    
    for country in countries:
        country_data = df[df['country'] == country].copy()
        country_code = country.replace(' ', '_').lower()
        
        # Ne traiter que les pays avec suffisamment de données
        if len(country_data) >= 30:  # Au moins 30 entrées
            output_file = f"data/by_country/{country_code}.csv"
            country_data.to_csv(output_file, index=False)
            print(f"Enregistré {len(country_data)} entrées pour {country} dans {output_file}")
    
    # Enregistrer l'ensemble complet des données prétraitées
    df.to_csv('data/processed/full_grouped_processed.csv', index=False)
    print(f"Données complètes prétraitées enregistrées dans data/processed/full_grouped_processed.csv")
    
    return True

def main():
    """
    Fonction principale exécutant le pipeline de préparation des données
    """
    print("=== DÉBUT DU PROCESSUS DE PRÉPARATION DES DONNÉES ===")
    print(f"Date et heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Étape 1: Chargement des données
    df = load_data()
    
    # Étape 2: Exploration des données
    df = explore_data(df)
    
    # Étape 3: Traitement des valeurs manquantes
    df = handle_missing_values(df)
    
    # Étape 4: Traitement des valeurs aberrantes
    df = handle_outliers(df)
    
    # Étape 5: Création de features dérivées
    df = create_derived_features(df)
    
    # Étape 6: Normalisation des données
    df_normalized = normalize_data(df)
    
    # Étape 7: Division des données
    split_data(df_normalized)
    
    print("\n=== FIN DU PROCESSUS DE PRÉPARATION DES DONNÉES ===")
    print(f"Date et heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nLes données sont prêtes pour l'entraînement des modèles d'apprentissage.")

if __name__ == "__main__":
    main()
