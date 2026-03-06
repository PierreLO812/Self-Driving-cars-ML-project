## Project: AI Failure Prediction in Autonomous Driving

## Présentation du Projet

Ce projet de Machine Learning vise à résoudre l'un des défis les plus critiques de la conduite autonome : déterminer quand l'IA n'est plus en mesure de conduire en toute sécurité et doit rendre la main à l'humain (Take-Over Request - TOR).

Plutôt que de simplement "mieux conduire", notre objectif est de construire une couche de "méta-intelligence" capable de détecter l'incapacité du système face à des conditions environnementales dégradées ou imprévues.

## Objectifs & Missions

1. Détection d'Incertitude (Uncertainty Estimation)

Utiliser des modèles probabilistes pour quantifier la confiance de l'IA. Si l'incertitude dépasse un seuil critique (ex: beaucoup d'objets visibles), le système déclenche une alerte.

2. Analyse de la Discrépance Expert/IA

Comparer la trajectoire prévue par le planificateur de nuPlan avec la trajectoire réelle effectuée par l'expert humain dans les mêmes conditions. Un écart significatif est un indicateur fort d'incapacité de l'IA.

3. Classification des Conditions Environnementales

Identifier automatiquement les facteurs de risque (nuit, pluie, trafic dense, zones de travaux) pour corréler les échecs de l'IA à des contextes spécifiques.

Stratégie de Données : Pourquoi nuPlan ?

Nous avons choisi le dataset nuPlan (par Motional) pour plusieurs raisons stratégiques liées à notre cours de Self-Driving Cars :

Réalisme du Planning : Contrairement aux datasets de vision pure, nuPlan se concentre sur la trajectoire, ce qui est l'étape finale avant l'action.

Diversité Géographique :

Singapour : Idéal pour tester la résilience face aux pluies tropicales et à la conduite à gauche.

Boston/Pittsburgh : Pour les environnements urbains complexes et les variations de luminosité.

Format SQLite : Permet de requêter efficacement des scénarios spécifiques (ex: "Extraire uniquement les virages à gauche sous la pluie") sans saturer la mémoire.

## Architecture Technique & Code

Stack Technologique

Langage : Python 3.9+

Framework ML : PyTorch (ou TensorFlow) pour les réseaux de neurones.

Data Handling : nuplan-devkit pour l'interface avec les bases de données .db.

Visualisation : Matplotlib & Plotly pour les cartes de chaleur d'incertitude.

Méthodologie de Développement

Phase d'Extraction (Data Engineering) : Filtrage des scènes via SQLite pour créer des sous-ensembles "Easy" (beau temps) vs "Hard" (pluie/nuit).

Modélisation :

Entraînement d'un modèle de prédiction de trajectoire.

Implémentation de couches de Dropout (Monte Carlo Dropout) pour estimer l'incertitude épistémique.

Évaluation : Création d'un score de "Safety Gap" (l'écart entre la décision IA et la sécurité minimale).

## Installation & Git Workflow

Installation

Cloner le repo : git clone (https://github.com/Pierre330ZB/ML-pj-self-drving-cars)

Installer le devkit : pip install nuplan-devkit

Configurer les chemins : Voir le fichier DATASET.md.

Workflow Git (Règles d'équipe)

Main : Branche de production, code stable uniquement.

Develop : Branche d'intégration des fonctionnalités.

Feature/[nom] : Branches individuelles pour chaque mission (ex: feature/uncertainty-model).

Interdiction : Ne jamais pusher de fichiers .db ou de dossiers data/ sur le repo (utilisez le .gitignore).

## Concepts Clés (Lien avec Coursera)

OOD (Out-of-Distribution) : Détecter quand l'environnement ne ressemble plus aux données d'entraînement.

LIDAR/Radar Fusion : Comprendre comment la perte d'un capteur influence la décision de rendre la main.

Minimal Risk Maneuver (MRM) : Le concept théorique de mise en sécurité du véhicule.

"Ce projet est réalisé dans le cadre de l'apprentissage des systèmes de conduite autonome de niveau 4."
