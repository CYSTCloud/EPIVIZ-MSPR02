"# EPIVIZ - Plateforme de Prédiction de Pandémies

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)
![Flask](https://img.shields.io/badge/flask-3.1.1-red.svg)
![ML](https://img.shields.io/badge/ML-sklearn%20%7C%20tensorflow-yellow.svg)

## Présentation

EPIVIZ est une plateforme avancée de prédiction et de visualisation des données de pandémies qui utilise des algorithmes d'apprentissage automatique pour générer des prévisions précises sur l'évolution des cas dans différents pays. Le projet intègre un pipeline de données complet, des modèles d'apprentissage automatique sophistiqués et une interface utilisateur intuitive pour explorer les prédictions.

## Table des matières

- [Installation](#-installation)
- [Architecture du projet](#-architecture-du-projet)
- [Pipeline de données](#-pipeline-de-données)
- [Modèles d'apprentissage](#-modèles-dapprentissage)
- [Interface utilisateur](#-interface-utilisateur)
- [API REST](#-api-rest)
- [Documentation](#-documentation)
- [Prochaines étapes](#-prochaines-étapes)

## Installation

### Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)
- Virtualenv (recommandé)

### Étapes d'installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/CYSTCloud/EPIVIZ-MSPR02.git
   cd EPIVIZ-MSPR02
   ```

2. **Créer un environnement virtuel** (optionnel mais recommandé)
   ```bash
   python -m venv venv
   # Sous Windows
   venv\Scripts\activate
   # Sous Linux/Mac
   source venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer l'application**
   ```bash
   python app.py
   ```
   L'application sera accessible à l'adresse http://localhost:8000

## Architecture du projet

Le projet EPIVIZ est structuré en plusieurs composants clés:

```
EPIVIZ-MSPR02/
├── data/                       # Données brutes et traitées
│   ├── by_country/             # Données séparées par pays
│   └── full_grouped.csv        # Données principales
├── models/                     # Modèles entraînés (.pkl, .keras)
├── results/                    # Résultats d'évaluation et visualisations
├── evaluation/                 # Métriques d'évaluation des modèles
├── templates/                  # Templates HTML pour l'interface web
├── static/                     # Fichiers statiques (CSS, JS)
├── documentation/              # Documentation technique
├── data_preparation.py         # Script de préparation des données
├── model_training.py           # Script d'entraînement des modèles
├── model_evaluation.py         # Script d'évaluation des performances
├── app.py                      # Application Flask et API REST
└── requirements.txt            # Dépendances du projet
```

## Pipeline de données

Le pipeline de données EPIVIZ est conçu pour traiter efficacement les données de pandémies en plusieurs étapes:

### 1. Chargement et exploration (data_preparation.py)

- Chargement des données à partir du fichier `full_grouped.csv`
- Division des données par pays
- Analyse exploratoire et visualisations statistiques

### 2. Prétraitement et feature engineering

- Conversion des dates en format datetime
- Traitement des valeurs manquantes et aberrantes
- Création de variables temporelles (jour de semaine, mois, année, etc.)
- Calcul de moyennes mobiles (7, 14, 30 jours)
- Calcul de taux de croissance
- Normalisation des données

### 3. Division des données

- Division chronologique (sans mélange aléatoire) en ensembles d'entraînement (80%) et de test (20%)
- Validation croisée temporelle pour l'optimisation des hyperparamètres

## Modèles d'apprentissage

EPIVIZ intègre plusieurs algorithmes d'apprentissage automatique pour les prédictions:

### Modèles classiques (model_training.py)

- **Linear Regression**: Modèle linéaire de base
- **Ridge Regression**: Régression avec régularisation L2
- **Lasso Regression**: Régression avec régularisation L1 (sélection de features)

### Modèles à base d'arbres

- **Random Forest**: Ensemble d'arbres de décision indépendants
- **Gradient Boosting**: Construction séquentielle d'arbres pour corriger les erreurs
- **XGBoost**: Implémentation optimisée du gradient boosting

### Modèles de deep learning

- **LSTM** (Long Short-Term Memory): Réseau de neurones récurrent adapté aux séries temporelles

### Optimisation des hyperparamètres

- Recherche par grille (GridSearchCV) avec validation croisée temporelle
- Optimisation des paramètres spécifiques à chaque type de modèle
- Métriques d'optimisation: RMSE, MAE, R²

## Interface utilisateur

L'interface utilisateur EPIVIZ est construite avec Flask et offre plusieurs fonctionnalités:

### Page d'accueil

- Liste des pays disponibles pour les prédictions
- Accès rapide aux fonctionnalités principales

### Page de prédiction par pays

- Sélection du nombre de jours à prédire (7-365)
- Choix du modèle à utiliser
- Visualisation interactive des prédictions (graphique)
- Tableau détaillé des valeurs prédites

### Page de comparaison entre pays

- Sélection de plusieurs pays à comparer
- Visualisation comparative des prédictions
- Analyse des tendances entre différentes régions

## API REST

L'API REST d'EPIVIZ permet d'intégrer les fonctionnalités de prédiction dans d'autres applications:

### Endpoints principaux

- `GET /api/countries`: Liste des pays disponibles
- `GET /api/predict/{country}?days=30&model=gradient_boosting`: Prédictions pour un pays
- `GET /api/historical/{country}`: Données historiques d'un pays
- `GET /api/compare?countries=Afghanistan,Brazil&days=30`: Comparaison entre pays

La documentation complète de l'API est disponible au format OpenAPI dans le fichier `documentation/api_documentation.yaml`.

## Documentation

Le projet inclut une documentation complète:

- **Guide d'utilisation**: Instructions détaillées pour les utilisateurs (`documentation/guide_utilisation.md`)
- **Documentation technique des modèles**: Architecture, hyperparamètres et performances (`documentation/documentation_technique_modeles.md`)
- **Documentation API**: Spécification OpenAPI des endpoints REST (`documentation/api_documentation.yaml`)

## Prochaines étapes

Le projet EPIVIZ pourrait être amélioré dans plusieurs directions:

- **Modèles plus avancés**: Intégration de modèles hiérarchiques, transformers ou bayésiens
- **Features additionnelles**: Données de mobilité, mesures sanitaires, vaccination
- **Méthodes d'ensemble**: Combinaison de plusieurs modèles pour des prédictions plus robustes
- **Déploiement en production**: Conteneurisation avec Docker et déploiement sur un service cloud
- **Interface mobile**: Développement d'une application mobile pour l'accès aux prédictions

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Contributeurs

- CYSTCloud - Développement et architecture

---
 
 2025 EPIVIZ - Plateforme de Prédiction de Pandémies
