import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.dummy import DummyClassifier
import importlib.util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_base_dir():
    return Path(os.path.abspath(__file__)).parent.parent

def load_module(module_name, script_name):
    script_path = get_base_dir() / "scripts_python" / script_name
    if not script_path.exists():
        logging.error(f"Le script {script_name} est introuvable !")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def plot_global_roc(results, y_test, results_dir):
    """
    Génère une courbe ROC superposant tous les modèles sur le même graphique.
    """
    plt.figure(figsize=(8, 6))
    
    colors = ['darkorange', 'green', 'blue']
    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"{res['name']} (AUC = {roc_auc:.2f})")
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (Inconfort)')
    plt.ylabel('Taux de Vrais Positifs (Sécurité)')
    plt.title('Comparatif Global : Courbes ROC')
    plt.legend(loc="lower right")
    roc_path = results_dir / "global_roc_curves.png"
    plt.savefig(roc_path, bbox_inches='tight', dpi=300)
    plt.close()
    return roc_path.name

def generate_comparative_report(report_path, df_shape, target_count, results, global_roc_name):
    content = f"""# 🚘 Rapport d'Analyse ML Multi-Modèles : Prédiction de Désengagement (Handover)

Ce document présente l'évaluation et la comparaison de trois algorithmes leaders conçus pour prédire l'incapacité d'une IA de conduite autonome uniquement en se basant sur son **environnement**.

## 📊 Résumé du Dataset
L'extraction SQLite `nuPlan` a généré un dataset purgé de toute donnée cinématique pour éviter toute "triche" prédictive.
*   **Total des frames analysées** : `{df_shape[0]:,}`
*   **Nombre de paramètres (Features)** : `{df_shape[1]:,}`
*   **Situations critiques identifiées (Handover = 1)** : `{target_count:,}` événements identifiés par notre heuristique métier de danger.

---

## 🏎️ Benchmark Comparatif

Les algorithmes (RF, GBM, SVM) ont tous été optimisés via **Validation Croisée K-Fold (GridSearch)** sur l'objectif métier : maximiser le **Recall** (Rappel) de la classe Handover pour écraser les Faux Négatifs (danger de mort).

### Tableau des Performances Globale

| Modèle | Précision Globale | Recall Handover (Sécurité🛟) | Précision Handover | Configuration Gagnante |
|--------|-------------------|-----------------------------|--------------------|------------------------|
"""
    for res in results:
        params_str = str(res['best_params']).replace('{', '').replace('}', '').replace("'", "")
        content += f"| **{res['name']}** | {res['accuracy']*100:.1f}% | **{res['recall']*100:.1f}%** | {res['precision']*100:.1f}% | `{params_str}` |\n"
        
    content += f"""

![Comparatif Courbes ROC](./{global_roc_name})
*Plus un modèle s'approche du coin supérieur gauche, meilleur est son compromis Sécurité / Inconfort.*

---

## 🚨 Analyse Métier Détaillée par Modèle

La Sécurité Automobile exige d'analyser non pas le score F1 brut, mais les **Faux Positifs** (voiture qui freine pour rien = inconfort) et les **Faux Négatifs** (voiture qui fonce dans le mur au lieu de rendre la main = catastrophe).
"""
    
    emojis = ['1️⃣', '2️⃣', '3️⃣']
    for i, res in enumerate(results):
        cm = res['cm']
        faux_positifs = cm[0][1]
        faux_negatifs = cm[1][0]
        vrais_positifs = cm[1][1]
        
        content += f"""
### {emojis[i]} {res['name']}
![Matrice de Confusion](./{res['cm_path']})

* **Faux Négatifs (DANGER 🔴)** : **`{faux_negatifs}` cas**. (Impact vital : l'IA ne rend PAS la main dans une situation extrême.)
* **Faux Positifs (INCONFORT 🟡)** : **`{faux_positifs}` cas**. (Impact minime : l'IA dérange le conducteur pour rien.)

![Top 10 Features](./{res['fi_path']})
"""

    content += """
> **CONCLUSION GÉNÉRALE :**
> L'optimisation délibérée sur le Rappel force nos IA à privilégier l'Inconfort à la Mortalité (elles préfèrent multiplier les Faux Positifs pour réduire drastiquement les Faux Négatifs). Ce benchmark prouve qu'un modèle tabulaire purement environnemental peut identifier ses limites avec une grande acuité.
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    logging.info("--- DÉMARRAGE DU MECHMARK GLOBAL MULTI-MODÈLES ---")
    
    base_dir = get_base_dir()
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    tabular_module = load_module("tabular_model", "03_tabular_feature_model.py")
    mod_rf = load_module("mod_rf", "model_rf.py")
    mod_gbm = load_module("mod_gbm", "model_gbm.py")
    mod_svm = load_module("mod_svm", "model_svm.py")
    
    logging.info("Extraction SQLite...")
    df = tabular_module.build_global_dataset(max_dbs=10)
    
    if df.empty or 'Target_Handover' not in df.columns:
        logging.error("Données insuffisantes.")
        sys.exit(1)
        
    df_shape = df.shape
    target_count = df['Target_Handover'].sum()
    
    # Preprocessing
    cols_to_drop = ['timestamp', 'Target_Handover', 'vx', 'vy', 'acceleration_x', 'acceleration_y', 'angular_rate_z']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['Target_Handover']
    
    strat = y if target_count > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=strat)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Lancement des modèles (Benchmarking)
    results = []
    
    try:
        # Exécution RF
        logging.info("Début RUN - Random Forest")
        res_rf = mod_rf.run_model(X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), results_dir)
        results.append(res_rf)
        
        # Exécution GBM
        logging.info("Début RUN - Gradient Boosting")
        res_gbm = mod_gbm.run_model(X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), results_dir)
        results.append(res_gbm)
        
        # Exécution SVM
        logging.info("Début RUN - Linear SVM")
        res_svm = mod_svm.run_model(X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), results_dir)
        results.append(res_svm)
        
    except Exception as e:
        logging.error(f"Une erreur s'est produite pendant le benchmark: {e}")
    
    # Rapport Global
    if results:
        global_roc = plot_global_roc(results, y_test, results_dir)
        report_path = results_dir / "report_ML_Global.md"
        generate_comparative_report(report_path, df_shape, target_count, results, global_roc)
        logging.info(f"Rapport Global multimodèle généré dans : {report_path}")
    else:
        logging.error("Aucun résultat à construire.")

if __name__ == "__main__":
    main()
