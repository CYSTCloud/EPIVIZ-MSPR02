# Documentation technique des modèles - EPIVIZ

## Introduction

Cette documentation technique détaille les modèles d'apprentissage automatique utilisés dans le projet EPIVIZ pour la prédiction des cas de pandémies. Elle couvre l'architecture des modèles, les techniques d'entraînement, les performances et les spécificités de chaque algorithme implémenté.

## Pipeline de données

### Prétraitement des données

Le prétraitement des données est géré par le script `data_preparation.py` qui implémente les transformations suivantes :

1. **Chargement des données** : Les données brutes sont chargées depuis le fichier `full_grouped.csv` et divisées par pays.
2. **Nettoyage** : 
   - Conversion des dates en format datetime
   - Traitement des valeurs manquantes par diverses méthodes (imputation par moyenne/médiane, valeurs précédentes, etc.)
   - Détection et traitement des valeurs aberrantes
3. **Feature engineering** :
   - Création de variables temporelles (jour de semaine, mois, année, jour, est_weekend, trimestre)
   - Calcul de moyennes mobiles (7 jours, 14 jours, 30 jours)
   - Calcul de taux de croissance
   - Normalisation de certaines features
   - Création de variables retardées (lag features)
4. **Division des données** :
   - Division chronologique (sans mélange aléatoire) en ensembles d'entraînement et de test (80%-20%)

### Principales features utilisées

| Catégorie | Features |
|-----------|----------|
| Données brutes | `new_cases`, `total_cases`, `new_deaths`, `total_deaths` |
| Temporelles | `year`, `month`, `day`, `day_of_week`, `is_weekend`, `quarter` |
| Dérivées | `*_ma7`, `*_ma14`, `*_ma30` (moyennes mobiles), `*_growth_rate` (taux de croissance), `*_normalized` (normalisées) |

## Modèles implémentés

Le script `model_training.py` entraîne plusieurs types de modèles pour chaque pays :

### 1. Modèles classiques

#### Linear Regression

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

- **Avantages** : Simple, interprétable, rapide à entraîner
- **Inconvénients** : Capacité limitée à capturer des relations non linéaires, sensible aux outliers
- **Hyperparamètres** : Aucun hyperparamètre à optimiser
- **Particularités** : Peut montrer des signes de surapprentissage avec un R² proche de 1.0

#### Ridge Regression

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=best_params['alpha'])
```

- **Avantages** : Régularisation L2 qui réduit le surapprentissage
- **Inconvénients** : Toujours limité aux relations linéaires
- **Hyperparamètres optimisés** : 
  - `alpha` : Contrôle la force de la régularisation (typiquement entre 0.1 et 10)

#### Lasso Regression

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=best_params['alpha'])
```

- **Avantages** : Régularisation L1 qui favorise la parcimonie (sélection de features)
- **Inconvénients** : Peut éliminer des variables importantes si mal calibré
- **Hyperparamètres optimisés** : 
  - `alpha` : Contrôle la force de la régularisation (typiquement entre 0.001 et 1)

### 2. Modèles basés sur les arbres

#### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
```

- **Avantages** : Robuste, capture les relations non linéaires, moins sensible au surapprentissage
- **Inconvénients** : Moins interprétable, peut être lourd en mémoire
- **Hyperparamètres optimisés** :
  - `n_estimators` : Nombre d'arbres (typiquement 100-500)
  - `max_depth` : Profondeur maximale des arbres (typiquement 10-30)
  - `min_samples_split` : Nombre minimum d'échantillons requis pour diviser un nœud
  - `min_samples_leaf` : Nombre minimum d'échantillons requis dans une feuille

#### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
```

- **Avantages** : Généralement très performant, construit les arbres séquentiellement pour corriger les erreurs
- **Inconvénients** : Sensible au surapprentissage, plus lent à entraîner
- **Hyperparamètres optimisés** :
  - `n_estimators` : Nombre d'arbres (typiquement 100-500)
  - `learning_rate` : Taux d'apprentissage qui contrôle la contribution de chaque arbre (0.01-0.2)
  - `max_depth` : Profondeur maximale des arbres (typiquement 3-10)
  - `min_samples_split` et `min_samples_leaf` : Contrôlent la taille minimale des nœuds

#### XGBoost

```python
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    min_child_weight=best_params['min_child_weight'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    random_state=42
)
```

- **Avantages** : Implémentation optimisée du gradient boosting, très performant
- **Inconvénients** : Plus complexe à configurer correctement
- **Hyperparamètres optimisés** :
  - `n_estimators` : Nombre d'arbres (typiquement 100-1000)
  - `learning_rate` : Taux d'apprentissage (0.01-0.3)
  - `max_depth` : Profondeur maximale des arbres (3-10)
  - `min_child_weight` : Somme minimale des poids des instances dans un enfant
  - `subsample` : Fraction des instances à utiliser pour chaque arbre (0.5-1.0)
  - `colsample_bytree` : Fraction des features à utiliser pour chaque arbre (0.5-1.0)

### 3. Modèle de deep learning

#### LSTM (Long Short-Term Memory)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Création des séquences pour LSTM
seq_length = 14
X_seq, y_seq = create_lstm_sequences(data_scaled, seq_length)

# Construction du modèle
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compilation
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement
model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32, 
          validation_split=0.2, callbacks=[early_stopping], verbose=1)
```

- **Avantages** : Conçu spécifiquement pour les séries temporelles, capture les dépendances à long terme
- **Inconvénients** : Nécessite plus de données, plus long à entraîner, sensible à l'initialisation
- **Architecture** :
  - Couche LSTM avec 50 unités et return_sequences=True
  - Dropout (20%) pour réduire le surapprentissage
  - Seconde couche LSTM avec 50 unités
  - Dropout (20%)
  - Couche de sortie Dense avec 1 unité
- **Hyperparamètres** :
  - Longueur de séquence : 14 jours
  - Batch size : 32
  - Epochs : 100 (avec early stopping)
  - Optimiseur : Adam
  - Fonction de perte : Mean Squared Error

## Processus d'optimisation des hyperparamètres

L'optimisation des hyperparamètres est réalisée à l'aide de `GridSearchCV` de scikit-learn :

```python
def optimize_hyperparameters(X_train, y_train, model_type):
    """Optimise les hyperparamètres pour un type de modèle spécifié"""
    
    if model_type == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
        
    elif model_type == 'gradient_boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        model = GradientBoostingRegressor(random_state=42)
        
    # [Autres modèles...]
    
    # Validation croisée adaptée aux séries temporelles
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Recherche par grille
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_
```

- **Méthode** : Recherche par grille (GridSearchCV)
- **Validation croisée** : TimeSeriesSplit avec 5 splits (respecte l'ordre chronologique)
- **Métrique d'optimisation** : Negative Mean Squared Error

## Performances des modèles

L'évaluation des modèles est réalisée dans le script `model_evaluation.py` qui calcule diverses métriques de performance.

### Résultats pour l'Afghanistan

| Modèle | MAE | RMSE | R² |
|--------|-----|------|---|
| Linear Regression | 1.02e-11 | 1.14e-11 | 1.0 |
| Gradient Boosting | 3.80 | 5.26 | 0.997 |
| XGBoost | 8.94 | 12.04 | 0.985 |
| Random Forest | 16.02 | 22.66 | 0.947 |
| LSTM | 69.16 | 83.40 | 0.264 |
| Lasso | 104.70 | 122.71 | -0.545 |
| Ridge | 110.40 | 129.27 | -0.715 |

### Analyse des résultats

1. **Linear Regression** : Le R² parfait (1.0) indique un probable surapprentissage. Le modèle a appris "par cœur" les données d'entraînement mais risque de mal généraliser.

2. **Gradient Boosting** : Excellent équilibre avec un R² de 0.997 et de faibles erreurs (MAE = 3.80, RMSE = 5.26). C'est le modèle le plus performant pour ce pays.

3. **XGBoost** : Très bonnes performances également (R² = 0.985), légèrement inférieures au Gradient Boosting.

4. **Random Forest** : Bonnes performances (R² = 0.947) mais erreurs plus élevées que les modèles de boosting.

5. **LSTM** : Performances limitées (R² = 0.264), ce qui suggère que l'architecture choisie ou la quantité de données disponibles n'étaient pas optimales pour ce modèle.

6. **Lasso et Ridge** : R² négatifs indiquant que ces modèles font pire qu'une simple prédiction par la moyenne. Ces modèles ne sont pas adaptés à ce cas d'usage.

### Importance des features

Pour les modèles basés sur les arbres (Random Forest, Gradient Boosting, XGBoost), l'importance des features a été analysée. Les variables les plus influentes sont généralement :

1. **new_cases_ma7** (moyenne mobile sur 7 jours) : Indicateur clé de la tendance récente
2. **total_cases** : Indicateur de la progression cumulative de l'épidémie
3. **day_of_week** : Capture les effets de saisonnalité hebdomadaire dans les rapports de cas

## Prédiction et déploiement

### Génération de prédictions futures

La génération de prédictions s'effectue de manière itérative :

1. Pour les modèles classiques :
   - On utilise les dernières données connues pour prédire le jour suivant
   - On met à jour les features avec la nouvelle prédiction
   - On répète pour le nombre de jours souhaité

2. Pour le modèle LSTM :
   - On utilise une fenêtre glissante de 14 jours
   - On prédit la valeur suivante
   - On ajoute cette prédiction à la fenêtre en supprimant la valeur la plus ancienne
   - On répète pour le nombre de jours souhaité

### API de prédiction

L'API Flask expose les modèles entraînés via plusieurs endpoints :

- `/api/predict/<country>?days=30&model=gradient_boosting` : Génère des prédictions pour un pays
- `/api/compare?countries=Afghanistan,Brazil&days=30` : Compare les prédictions entre pays

Les prédictions sont renvoyées au format JSON et peuvent être visualisées via l'interface web.

## Limites et perspectives d'amélioration

### Limites actuelles

1. **Mise à jour des variables dérivées** : Lors de la génération de prédictions séquentielles, certaines variables dérivées (comme les moyennes mobiles) ne sont pas correctement mises à jour, ce qui peut affecter la qualité des prédictions à long terme.

2. **Horizon de prédiction** : La fiabilité des prédictions diminue généralement avec l'allongement de l'horizon temporel.

3. **Validation croisée** : Bien que nous utilisions TimeSeriesSplit, une validation plus rigoureuse avec plusieurs fenêtres temporelles pourrait améliorer la robustesse des modèles.

4. **Spécificités des pays** : Les modèles sont entraînés indépendamment pour chaque pays, sans tenir compte des similitudes ou des interactions entre pays.

### Perspectives d'amélioration

1. **Modèles plus avancés** : 
   - Modèles hiérarchiques qui capturent les relations entre pays
   - Architectures de deep learning plus sophistiquées (réseaux convolutifs temporels, transformers)
   - Modèles bayésiens pour quantifier l'incertitude des prédictions

2. **Features additionnelles** :
   - Données de mobilité
   - Mesures de restrictions sanitaires
   - Données de vaccination
   - Variables météorologiques

3. **Méthodes d'ensemble** : Combiner les prédictions de plusieurs modèles pour améliorer la robustesse et la précision.

4. **Réentraînement automatique** : Mettre en place un système de réentraînement périodique des modèles avec les nouvelles données disponibles.

## Conclusion

Les modèles implémentés dans le projet EPIVIZ offrent des capacités de prédiction fiables, particulièrement avec le Gradient Boosting qui démontre d'excellentes performances. L'architecture modulaire du projet permet d'ajouter facilement de nouveaux modèles ou d'améliorer les existants.

L'évaluation rigoureuse des performances et l'analyse des importances de features fournissent des informations précieuses sur les facteurs qui influencent l'évolution des pandémies et permettent de guider les décisions en matière de santé publique.
