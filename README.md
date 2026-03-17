🚘 Couche de Superviseur ML : Prédiction du Transfert de Contrôle (Handover)

📌 Vision du Projet

Dans le développement des véhicules autonomes de Niveau 4, la question n'est plus seulement de savoir si la voiture peut conduire, mais de savoir quand elle ne peut plus le faire.

Ce projet ne propose pas un modèle de conduite, mais une couche de méta-intelligence (superviseur). Son rôle est d'analyser l'environnement en temps réel pour prédire si le système de conduite autonome va atteindre ses limites de sécurité, nécessitant ainsi de rendre la main (Handover) au conducteur humain.

🎯 Pourquoi ce projet est-il crucial ?

Un véhicule autonome fait face à un dilemme permanent entre deux impératifs :

L'Impératif de Sécurité : Ne jamais se retrouver dans une situation que l'IA ne sait pas gérer (éviter les accidents).

L'Impératif de Confort : Ne pas interrompre la conduite autonome sans raison valable (éviter la frustration de l'utilisateur).

Notre travail consiste à trouver le réglage mathématique idéal pour que la voiture soit prudente sans être paranoïaque.

🧠 Fonctionnement de l'Approche

Plutôt que d'analyser la trajectoire de la voiture (vitesse, angle de volant), nous nous concentrons exclusivement sur son contexte environnemental.

1. Analyse de la complexité

Nous extrayons des données du dataset nuPlan (Singapore train) pour identifier les facteurs de risque :

Densité de trafic : Nombre de piétons et de véhicules aux alentours.

Complexité de la scène : Présence d'intersections, de panneaux stop ou de zones de travaux.

Conditions géographiques : Conduite à gauche (Singapour) vs conduite à droite (Boston).

2. Détection d'incapacité (Prédiction Anticipatoire à +2 Frames)

Le système apprend à corréler ces facteurs environnementaux avec les moments où un conducteur expert a jugé nécessaire de reprendre les commandes. 
**Innovation technique :** Notre modèle ne se contente pas de réagir au moment présent. La valeur cible est décalée virtuellement (Time Shift). Ainsi, à l'instant `t`, le modèle est entraîné pour prédire si la situation sera critique à l'instant `t+2`. Cette **anticipation de 2 frames** offre au véhicule une marge précieuse pour initier le transfert (Handover) avec souplesse, ou engager un freinage de sécurité (Minimal Risk Maneuver) avant l'impact.

🛠️ Méthodologie et Stratégie ML

Nous avons exploré plusieurs philosophies de modèles pour répondre aux exigences du barème :

L'Approche Sécuritaire (Random Forest) : Un modèle conçu pour ne rater aucun danger. Il est extrêmement sûr mais a tendance à rendre la main trop souvent dès qu'un doute subsiste.

L'Approche Équilibrée (Boosting) : Un modèle plus fin qui cherche à mieux distinguer les situations réellement critiques des situations simplement denses.

Le Modèle Hybride (Final) : Une combinaison intelligente (Voting) qui permet d'ajuster le curseur de décision. Cela permet de garantir qu'une fois le système déployé en production, on puisse choisir de privilégier soit la fluidité, soit la sécurité absolue.

🎓 Concepts Clés (Lien avec le cours Coursera)

Ce projet met en pratique des piliers théoriques de la conduite autonome :

OOD (Out-of-Distribution) : Reconnaître quand la voiture entre dans un scénario qu'elle n'a jamais rencontré à l'entraînement.

Minimal Risk Maneuver (MRM) : Le handover est le point de départ de la mise en sécurité du véhicule.

Interprétabilité : Comprendre quels objets (ex: cyclistes vs feux rouges) causent le plus de stress au système de planification.

🚀 Structure du Repository

*   `scripts_python/` : Extraction des scènes depuis SQLite, nettoyage, et entraînement des modèles (RF, GBM, Hybride).
*   `hardware_control_test/` : **Déploiement Physique (C++ & Computer Vision)**. L'objectif ultime étant l'implémentation dans un véhicule réel, nous avons inclus des algorithmes C++ pour contrôler une voiture miniature télécommandée. Un module OpenCV autonome (`vision_controller.cpp`) analyse un flux caméra en temps réel, extrait les lignes de route, et génère des consignes de direction (Steering/Throttle).


Entraînement et optimisation des algorithmes de classification.

main.py : Script principal orchestrant tout le pipeline, du chargement des données à l'évaluation finale.

Ce projet s'inscrit dans une démarche de recherche sur la fiabilité des systèmes autonomes complexes et a pour objectif un rendu académique, il a été crée dans pour une utilisation dans un cadre scolaire.
Les dataasets étant très volumineux, ces derniers ne sont pas déposés dans le Repo présent ici (70Go).
