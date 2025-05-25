# Guide d'utilisation - EPIVIZ Prédiction de Pandémies

## Introduction

EPIVIZ est une plateforme avancée de prédiction des pandémies qui utilise des modèles d'apprentissage automatique pour générer des prévisions précises concernant l'évolution des cas de maladies infectieuses. Cette application permet aux utilisateurs de visualiser les prédictions pour différents pays, comparer des tendances entre pays et accéder aux données historiques.

Ce guide explique en détail comment utiliser l'application EPIVIZ, ses fonctionnalités principales et comment interpréter les résultats.

## Configuration requise

- Un navigateur web moderne (Chrome, Firefox, Edge, Safari)
- Python 3.10+ avec les dépendances installées (voir requirements.txt)
- 8 Go de RAM minimum recommandés pour l'exécution des modèles

## Installation et démarrage

1. **Installation des dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Démarrage de l'application** :
   ```bash
   python app.py
   ```

3. **Accès à l'interface web** :
   Ouvrez votre navigateur et accédez à http://localhost:8000

## Fonctionnalités principales

### 1. Page d'accueil

La page d'accueil présente la liste des pays pour lesquels des modèles de prédiction sont disponibles. Vous pouvez :

- Parcourir la liste des pays disponibles
- Cliquer sur un pays pour accéder à sa page de prédiction
- Accéder à la page de comparaison via le menu de navigation

### 2. Page de prédiction par pays

Cette page vous permet de générer et visualiser des prédictions pour un pays spécifique.

#### Fonctionnalités :

- **Sélection des paramètres de prédiction** :
  - Nombre de jours à prédire (entre 7 et 365)
  - Modèle à utiliser (le meilleur modèle est sélectionné par défaut)

- **Visualisation des résultats** :
  - Graphique interactif montrant les données historiques et les prédictions
  - Tableau détaillé des prédictions jour par jour

#### Comment utiliser cette page :

1. Ajustez les paramètres selon vos besoins
2. Cliquez sur "Générer les prédictions"
3. Analysez le graphique qui s'affiche :
   - Ligne bleue : données historiques
   - Ligne rouge pointillée : prédictions
   - La ligne verticale verte marque la séparation entre les données historiques et les prédictions
4. Consultez le tableau pour voir les valeurs précises pour chaque jour prédit

### 3. Page de comparaison entre pays

Cette page permet de comparer les prédictions entre plusieurs pays.

#### Fonctionnalités :

- **Sélection des paramètres** :
  - Sélection de plusieurs pays à comparer (minimum 2)
  - Nombre de jours à prédire (entre 7 et 90)

- **Visualisation comparative** :
  - Graphique interactif avec une courbe distincte pour chaque pays
  - Légende permettant d'identifier chaque pays

#### Comment utiliser cette page :

1. Cochez les cases correspondant aux pays que vous souhaitez comparer
2. Spécifiez le nombre de jours pour la prédiction
3. Cliquez sur "Comparer les pays"
4. Analysez le graphique comparatif :
   - Survolez les points pour voir les valeurs précises
   - Cliquez sur les éléments de la légende pour masquer/afficher certains pays

## Interprétation des résultats

### Modèles disponibles

L'application EPIVIZ utilise plusieurs modèles d'apprentissage automatique, chacun avec ses forces et faiblesses :

- **Gradient Boosting** : Généralement le plus performant, avec une bonne capacité à capturer les tendances complexes
- **XGBoost** : Très efficace pour les prédictions à court terme, gère bien les données bruitées
- **Random Forest** : Robuste et fiable, moins susceptible au surapprentissage
- **Linear Regression, Ridge, Lasso** : Modèles linéaires simples, utilisés comme référence
- **LSTM** : Modèle de réseau de neurones spécialisé pour les séries temporelles, performant pour capturer les dépendances à long terme

### Métriques d'évaluation

Pour évaluer la qualité des prédictions, plusieurs métriques sont utilisées :

- **MAE (Mean Absolute Error)** : Erreur moyenne absolue, mesure l'écart moyen entre les prédictions et les valeurs réelles
- **RMSE (Root Mean Square Error)** : Racine carrée de l'erreur quadratique moyenne, pénalise davantage les erreurs importantes
- **R² (coefficient de détermination)** : Mesure la proportion de la variance expliquée par le modèle (plus proche de 1 est meilleur)

### Limitations des prédictions

Il est important de comprendre les limitations des prédictions :

1. **Horizon de prédiction** : La fiabilité diminue généralement à mesure que l'horizon de prédiction s'allonge
2. **Événements imprévus** : Les modèles ne peuvent pas prédire des événements exceptionnels ou des changements de politique sanitaire
3. **Qualité des données** : Les prédictions dépendent de la qualité et de la complétude des données historiques
4. **Variations entre modèles** : Différents modèles peuvent produire des résultats différents pour le même pays

## API REST

L'application expose également une API REST pour l'intégration avec d'autres systèmes :

- **GET /api/countries** : Liste des pays disponibles
- **GET /api/predict/{country}** : Prédictions pour un pays spécifique
- **GET /api/historical/{country}** : Données historiques d'un pays
- **GET /api/compare** : Comparaison entre plusieurs pays

Pour plus de détails sur l'API, consultez la documentation OpenAPI dans le fichier `documentation/api_documentation.yaml`.

## Dépannage

### Problèmes courants

1. **L'application ne démarre pas** :
   - Vérifiez que toutes les dépendances sont installées : `pip install -r requirements.txt`
   - Assurez-vous qu'aucune autre application n'utilise le port 8000

2. **Les prédictions ne s'affichent pas** :
   - Vérifiez la connexion internet (certaines ressources JavaScript sont chargées depuis des CDN)
   - Essayez de rafraîchir la page
   - Consultez les logs du serveur Flask pour identifier d'éventuelles erreurs

3. **Les graphiques ne sont pas interactifs** :
   - Assurez-vous que JavaScript est activé dans votre navigateur
   - Essayez un autre navigateur pour isoler le problème

## Assistance

Pour toute question ou problème concernant l'application EPIVIZ, consultez la documentation technique ou contactez l'équipe de support.
