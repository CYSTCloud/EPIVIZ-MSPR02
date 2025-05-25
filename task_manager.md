# EPIVIZ PREDICTIONS - Task Manager (Mis à jour)

> Ce task manager est structuré pour couvrir toutes les compétences évaluées dans la grille E6.2 DEVIA FS et garantir un niveau de maîtrise 3 pour chaque critère. Il répond également à tous les livrables attendus selon l'expression des besoins.

## Problèmes rencontrés et leçons apprises

### Difficultés techniques identifiées

1. **Intégration des modèles préentraînés dans l'API** :
   - Problèmes de chargement et d'accès aux modèles enregistrés
   - Difficultés avec les chemins de fichiers et la structure du projet
   - Problèmes d'initialisation des services au démarrage de l'API

2. **Gestion des données** :
   - Incohérences dans les noms de colonnes entre les différents fichiers (`date` vs `date_value`)
   - Difficultés d'accès aux données historiques pour certains pays

3. **Architecture de l'API** :
   - Complexité excessive dans certaines parties du code
   - Manque de robustesse dans les mécanismes de fallback et de gestion d'erreurs
   - Dépendances circulaires et problèmes d'importation

### Approche révisée

1. **Structuration par étapes** :
   - Adopter une approche progressive avec tests intermédiaires
   - Vérifier chaque composant individuellement avant l'intégration
   - Documentation précise des interfaces entre composants

2. **Gestion des modèles** :
   - Standardisation des interfaces de modèles
   - Mécanisme de chargement robuste avec alternatives clairement définies
   - Tests unitaires pour chaque type de modèle

3. **Assurance qualité** :
   - Tests de non-régression après chaque étape majeure
   - Centralisation des logs et amélioration du diagnostic
   - Mise en place de mécanismes de récupération multi-niveaux


## Livrables attendus

### Modèles d'apprentissage
- [ ] Modèles entraînés par pays pour la prédiction des cas de pandémie (fichiers .pkl)
- [ ] Documentation détaillée sur le choix des algorithmes d'apprentissage et leurs performances
- [ ] Analyse comparative des différents modèles testés (résultats d'évaluation)
- [ ] Scripts Python pour l'entraînement et l'évaluation des modèles

### Interface et visualisation
- [ ] Interface web minimale pour la sélection de pays et l'affichage des prédictions
- [ ] Graphiques comparant les prédictions aux valeurs réelles
- [ ] Documentation sur les principes d'ergonomie/accessibilité implémentés

### Documentation technique
- [ ] Description du processus de préparation des données
- [ ] Explication des features utilisées pour l'entraînement
- [ ] Justification des hyperparams des modèles et méthodes d'optimisation
- [ ] Benchmark des solutions Front-end
- [ ] Application Front-end moderne avec justification des technologies
- [ ] API IA développée en Python avec justification des technologies
- [ ] Documentation d'API type OPEN API
- [ ] Tests automatisés et rapport de couverture pour l'interface utilisateur
- [ ] Mémoire méthodologique et analyse technique

## Tâches prioritaires pour l'entraînement des modèles

### 1. Préparation des données
- [ ] Exploration approfondie du fichier full_grouped.csv
- [ ] Détection et traitement des valeurs manquantes et aberrantes
- [ ] Création de features dérivées (moyennes mobiles, taux de croissance, etc.)
- [ ] Normalisation des données numériques
- [ ] Division en ensembles d'entraînement/validation/test

### 2. Sélection et entraînement des modèles
- [ ] Tester plusieurs algorithmes (régression, forêts aléatoires, gradient boosting, réseaux de neurones)
- [ ] Optimiser les hyperparamètres pour chaque type de modèle
- [ ] Implémenter des techniques spécifiques aux séries temporelles (LSTM, ARIMA)
- [ ] Entraîner des modèles spécifiques par pays avec les données les plus pertinentes
- [ ] Sauvegarder les modèles entraînés (.pkl) pour réutilisation

### 3. Évaluation des performances
- [ ] Implémenter une cross-validation adaptée aux séries temporelles
- [ ] Calculer les métriques pertinentes (MAE, RMSE, R²)
- [ ] Analyser les erreurs et identifier les zones d'amélioration
- [ ] Comparer les performances des différents modèles
- [ ] Générer des visualisations (prédictions vs valeurs réelles)

### 4. Développement de l'interface
- [ ] Créer une API Flask pour exposer les prédictions
- [ ] Développer un frontend minimaliste pour la sélection de pays
- [ ] Implémenter des graphiques interactifs pour visualiser les prédictions
- [ ] Ajouter des fonctionnalités de comparaison entre pays couverture pour l'interface utilisateur
- [ ] Documentation sur la conduite au changement (accessibilité)
- [ ] Gestion de projet évolutive suivant les méthodes agiles

## Phase 1: Préparation de l'environnement et analyse

> **Compétence évaluée**: Paramétrer un environnement de codage (Framework) adéquat pour développer le modèle d'apprentissage

### 1.1 Configuration de l'environnement de développement (Maîtrise des connaissances associées)
- [ ] Créer un environnement virtuel Python (`python -m venv venv`)
- [ ] Activer l'environnement virtuel (`source venv/bin/activate` ou `venv\Scripts\activate` sur Windows)
- [ ] Installer les dépendances nécessaires:
  - [ ] FastAPI (`pip install fastapi`)
  - [ ] Uvicorn (`pip install uvicorn`)
  - [ ] Pandas (`pip install pandas`)
  - [ ] Scikit-learn (`pip install scikit-learn`)
  - [ ] Statsmodels ou Prophet si nécessaire (`pip install statsmodels` ou `pip install prophet`)
  - [ ] Autres bibliothèques utiles (`pip install python-multipart pydantic`)
- [ ] Créer un fichier requirements.txt (`pip freeze > requirements.txt`)
- [ ] Documenter les choix technologiques et leurs justifications

### 1.2 Structuration du projet
- [ ] Créer la structure de dossiers:
  ```
  EPIVIZ_PREDICTIONS/
  ├── app/
  │   ├── __init__.py
  │   ├── main.py
  │   ├── data/
  │   │   ├── __init__.py
  │   │   └── data_service.py
  │   ├── models/
  │   │   ├── __init__.py
  │   │   └── prediction_model.py
  │   ├── api/
  │   │   ├── __init__.py
  │   │   └── routes.py
  │   └── static/
  │       ├── index.html
  │       ├── css/
  │       └── js/
  ├── data/
  │   └── full_grouped.csv
  ├── models/
  │   ├── 
  ├── requirements.txt
  └── README.md
  ```
- [ ] Initialiser Git si nécessaire (`git init`)

## Phase 2: Développement du service d'accès aux données

> **Compétence évaluée**: Générer des données d'entrée, récolter et adapter les types de données traitées nécessaires au modèle d'apprentissage en utilisant des approches et des outils adaptés

### 2.1 Création du service de données (Cohérence des approches et outils de préparation)
- [ ] Créer le module `data_service.py` pour lire les fichiers CSV
- [ ] Implémenter les fonctions d'accès aux données:
  - [ ] `get_countries()`: retourne la liste des pays disponibles
  - [ ] `get_historical_data(country)`: retourne les données historiques pour un pays
  - [ ] `get_country_data_for_prediction(country)`: prépare les données pour la prédiction

### 2.2 Adaptation des données pour le modèle (Cohérence de l'adaptation par rapport au modèle)
- [ ] Implémenter des fonctions de prétraitement des données temporelles:
  - [ ] Normalisation/Standardisation si nécessaire
  - [ ] Création de variables décalées (lag features) pour l'apprentissage
  - [ ] Gestion des valeurs extrêmes ou manquantes
- [ ] Documenter le processus d'adaptation des données pour le modèle

### 2.3 Validation des données (Fiabilité - Propres - Sécurisées)
- [ ] Tester la fonction `get_countries()` pour vérifier qu'elle retourne bien la liste des pays
- [ ] Tester la fonction `get_historical_data()` avec différents pays pour vérifier la cohérence des données
- [ ] Vérifier la performance des fonctions d'accès aux données
- [ ] Implémenter des validations pour garantir l'intégrité des données
- [ ] Documenter les vérifications de fiabilité et sécurité des données

## Phase 3: Développement du modèle de prédiction

> **Compétence évaluée**: Coder le modèle d'apprentissage choisi en maîtrisant les différentes architectures dans un environnement de développement

### 3.1 Analyse et choix du modèle (Maîtrise des différentes architectures)
- [ ] Analyser les caractéristiques des données temporelles (tendances, saisonnalité)
- [ ] Comparer différentes architectures de modèles:
  - [ ] Modèles statistiques (ARIMA/SARIMA)
  - [ ] Modèles à base d'arbres (Random Forest, Gradient Boosting)
  - [ ] Réseaux de neurones (si pertinent)
  - [ ] Modèles spécialisés pour séries temporelles (Prophet)
- [ ] Justifier le choix de l'architecture retenue par rapport au problème
- [ ] Définir la fenêtre temporelle pour les prédictions (7 jours par défaut)

### 3.2 Implémentation du modèle (Maîtrise de l'environnement de développement)
- [ ] Développer le module `prediction_model.py` avec les classes/fonctions de prédiction
- [ ] Implémenter la classe principale du modèle avec une architecture modulaire
- [ ] Développer une interface cohérente pour différentes méthodes de prédiction
- [ ] Prévoir un mécanisme pour sauvegarder/charger les modèles entraînés (.pkl)
- [ ] Mettre en place des tests unitaires pour vérifier le fonctionnement du modèle

### 3.3 Paramétrage du modèle (Maîtrise du paramétrage)
- [ ] Identifier les hyperparamètres clés du modèle choisi
- [ ] Implémenter une méthode de recherche d'hyperparamètres (Grid Search ou Random Search)
- [ ] Documenter l'impact des différents paramètres sur les performances
- [ ] Mettre en place une configuration modulaire des paramètres

### 3.4 Qualité du code et opérationnalité
- [ ] Suivre les principes SOLID dans l'implémentation
- [ ] Documenter le code avec des docstrings clairs
- [ ] Implémenter la gestion des erreurs et exceptions
- [ ] Vérifier l'opérationnalité complète du modèle (prédictions fonctionnelles)

## Phase 4: Développement de l'API FastAPI

### 4.1 Configuration de l'application FastAPI
- [ ] Créer le fichier `main.py` avec l'instance FastAPI
- [ ] Configurer CORS pour permettre les appels depuis le frontend
- [ ] Ajouter le montage des fichiers statiques

### 4.2 Implémentation des routes API
- [ ] Créer le module `routes.py` pour définir les endpoints
- [ ] Implémenter l'endpoint `/api/countries` pour lister les pays
- [ ] Implémenter l'endpoint `/api/historical/{country}` pour les données historiques
- [ ] Implémenter l'endpoint `/api/predict/{country}` pour les prédictions
- [ ] Ajouter la validation des paramètres et la gestion des erreurs

### 4.3 Test des endpoints API
- [ ] Tester l'endpoint `/api/countries` pour vérifier le format de la réponse
- [ ] Tester l'endpoint `/api/historical/{country}` avec différents pays
- [ ] Tester l'endpoint `/api/predict/{country}` avec différentes valeurs de jours
- [ ] Vérifier la gestion des erreurs (pays inconnu, paramètres invalides)

## Phase 5: Développement du frontend

### 5.1 Création de l'interface utilisateur
- [ ] Développer le fichier `index.html` avec la structure de base
- [ ] Créer le fichier CSS pour le style de l'interface
- [ ] Créer le fichier JavaScript pour l'interactivité

### 5.2 Intégration avec l'API
- [ ] Implémenter la fonction pour charger la liste des pays
- [ ] Implémenter la fonction pour récupérer les données historiques
- [ ] Implémenter la fonction pour récupérer les prédictions
- [ ] Gérer les états de chargement et les erreurs

### 5.3 Benchmark et choix de la solution de visualisation
- [ ] Effectuer un benchmark des bibliothèques de visualisation (Chart.js, D3.js, Plotly)
- [ ] Comparer les fonctionnalités, la performance et la facilité d'intégration
- [ ] Documenter la comparaison et justifier le choix de la bibliothèque
- [ ] Intégrer la bibliothèque sélectionnée

### 5.4 Visualisation des données
- [ ] Implémenter l'affichage des données historiques
- [ ] Implémenter l'affichage des prédictions
- [ ] Ajouter des légendes et explications pour l'utilisateur
- [ ] Implémenter des options d'accessibilité (contraste, textes alternatifs, navigation clavier)

## Phase 5: Procédure d'entraînement et ajustement du modèle

> **Compétence évaluée**: Réaliser et paramétrer une procédure d'entraînement adéquate d'un modèle d'apprentissage et ajuster l'apprentissage du modèle à partir du taux d'apprentissage et des résultats obtenus

### 5.1 Paramétrage de la procédure d'entraînement
- [ ] Implémenter différentes stratégies d'entraînement:
  - [ ] Entraînement par lot (batch training)
  - [ ] Entraînement incrémental si applicable
- [ ] Sélectionner les données d'entraînement les plus pertinentes
- [ ] Configurer les paramètres d'entraînement (nombre d'itérations, taux d'apprentissage)
- [ ] Documenter la procédure d'entraînement choisie

### 5.2 Exécution de l'entraînement
- [ ] Créer un script pour entraîner les modèles pour chaque pays (ou les principaux)
- [ ] Implémenter un mécanisme de logging pour suivre l'évolution de l'entraînement
- [ ] Sauvegarder les modèles entraînés dans le dossier `models/`
- [ ] Tester le chargement des modèles sauvegardés

### 5.3 Ajustement du modèle basé sur les résultats
- [ ] Analyser les performances initiales du modèle
- [ ] Identifier les points faibles et les opportunités d'amélioration
- [ ] Ajuster les hyperparamètres en fonction des résultats (taux d'apprentissage, complexité)
- [ ] Réentraîner le modèle avec les paramètres ajustés
- [ ] Comparer les performances avant et après ajustement
- [ ] Documenter les améliorations obtenues

## Phase 6: Tests et validation

> **Compétence évaluée**: Réaliser une phase de test en choisissant une méthode appropriée afin d'analyser la performance du modèle de données

### 6.1 Tests automatisés du frontend
- [ ] Mettre en place l'environnement de test (Jest, Cypress)
- [ ] Développer des tests unitaires pour les composants frontend
- [ ] Développer des tests d'intégration pour les interactions avec l'API
- [ ] Développer des tests end-to-end pour les fonctionnalités complètes
- [ ] Générer un rapport de couverture des tests

### 6.2 Méthodologie de test du modèle (Choix de méthode appropriée)
- [ ] Implémenter la validation croisée temporelle (Time Series Cross Validation)
- [ ] Mettre en place une stratégie de test cohérente avec le contexte temporel
- [ ] Documenter la méthodologie de test choisie et sa justification
- [ ] Comparer avec d'autres méthodes possibles (Bootstrap, Hold-out simple)

### 6.3 Définition des indicateurs de performance
- [ ] Implémenter les métriques d'évaluation:
  - [ ] Erreur moyenne absolue (MAE) et pourcentage (MAPE)
  - [ ] Erreur quadratique moyenne (RMSE)
  - [ ] Coefficient de détermination (R²)
  - [ ] Pour les aspects classification (si applicable): précision, rappel, F1-score
- [ ] Créer des visualisations pour analyser les erreurs de prédiction
- [ ] Documenter l'interprétation des différentes métriques

### 6.4 Analyse de la performance
- [ ] Analyser le taux d'apprentissage et la convergence du modèle
- [ ] Évaluer les performances sur différents pays et périodes
- [ ] Identifier les forces et faiblesses du modèle
- [ ] Créer un rapport d'analyse de performance

### 6.5 Tests d'intégration système
- [ ] Tester le flux complet (sélection de pays → affichage des données → prédictions)
- [ ] Vérifier que les prédictions du système en ligne correspondent aux résultats attendus
- [ ] Tester sur différents navigateurs (Chrome, Firefox)
- [ ] Mesurer et optimiser les temps de réponse
- [ ] Tester l'accessibilité avec des outils comme axe ou Lighthouse

## Phase 7: Documentation et finalisation

> **Compétence transversale**: Qualité du code, documentation, et professionnalisme

### 7.1 Nettoyage et optimisation du code
- [ ] Réviser le code pour la lisibilité et la maintenabilité
- [ ] Éliminer le code redondant ou inutilisé
- [ ] Refactoriser le code selon les principes SOLID
- [ ] Ajouter des commentaires où nécessaire
- [ ] Vérifier que le code respecte les conventions PEP 8

### 7.2 Documentation OPEN API
- [ ] Intégrer Swagger UI à l'API FastAPI
- [ ] Configurer les descriptions et exemples pour chaque endpoint
- [ ] Générer la documentation OpenAPI complète (JSON/YAML)
- [ ] Tester l'interaction avec la documentation API depuis l'interface Swagger

### 7.3 Documentation technique complète
- [ ] Rédiger une documentation détaillée sur le choix de l'algorithme IA
- [ ] Documenter les principes d'ergonomie et d'accessibilité mis en place
- [ ] Justifier les technologies utilisées (frontend et backend)
- [ ] Mettre à jour le README.md avec les instructions d'installation et d'utilisation
- [ ] Rédiger une documentation technique détaillant:
  - [ ] L'architecture du système
  - [ ] Les performances du modèle et les métriques obtenues
  - [ ] Les limitations connues et pistes d'amélioration

### 7.4 Documentation de conduite au changement (accessibilité)
- [ ] Documenter les fonctionnalités d'accessibilité implémentées
- [ ] Élaborer des guides pour les utilisateurs avec besoins spécifiques
- [ ] Proposer un plan d'évolution pour améliorer l'accessibilité
- [ ] Faire référence aux normes WCAG 2.1

### 7.5 Préparation au déploiement
- [ ] Configurer les paramètres de production pour FastAPI
- [ ] Préparer les scripts de démarrage
- [ ] Mettre en place des configurations sécurisées
- [ ] Tester l'application en mode production localement

## Phase 8: Déploiement

### 8.1 Configuration de l'environnement de déploiement
- [ ] Préparer l'environnement de déploiement (local ou conteneurisé)
- [ ] Configurer les variables d'environnement
- [ ] Mettre en place un processus de déploiement automatisé

### 8.2 Déploiement de l'application
- [ ] Déployer l'application FastAPI
- [ ] Vérifier l'accès aux fichiers CSV et aux modèles entraînés
- [ ] Tester l'application déployée

### 8.3 Tests post-déploiement
- [ ] Vérifier que tous les endpoints fonctionnent correctement
- [ ] Vérifier que l'interface utilisateur est accessible et fonctionnelle
- [ ] Vérifier que les prédictions sont générées correctement
- [ ] Vérifier la conformité aux normes d'accessibilité

## Phase 9: Gestion de projet et évolution

### 9.1 Mise en place d'une gestion de projet agile
- [ ] Définir les sprints et les user stories pour les développements futurs
- [ ] Créer un backlog de fonctionnalités pour les prochaines itérations
- [ ] Établir un processus de suivi et de reporting pour les avancées
- [ ] Préparer un plan d'évolution du projet en fonction des besoins de l'OMS

### 9.2 Plan d'évolution et d'amélioration continue
- [ ] Identifier les opportunités d'amélioration du modèle
- [ ] Proposer un plan pour l'intégration de nouvelles sources de données
- [ ] Planifier l'évolution des fonctionnalités de l'interface utilisateur
- [ ] Définir des indicateurs de suivi pour mesurer l'amélioration continue

## Phase 10: Présentation et évaluation

### 10.1 Préparation de la présentation
- [ ] Préparer une démonstration complète de l'application
- [ ] Documenter les compétences démontrées selon la grille d'évaluation
- [ ] Mettre en avant les choix technologiques et leur justification
- [ ] Préparer les réponses aux questions potentielles

### 10.2 Évaluation finale
- [ ] Vérifier que tous les critères d'évaluation sont satisfaits
- [ ] S'assurer que tous les livrables attendus sont complétés
- [ ] Vérifier que l'application est stable, performante et accessible
- [ ] Préparer un résumé des choix techniques et des performances
