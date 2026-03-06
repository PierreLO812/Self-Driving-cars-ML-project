import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def run_model(X_train, X_test, y_train, y_test, features_names, results_dir):
    logging.info("--- Entraînement Modèle Hybride (RF + GBM) ---")
    
    # Paramètres gagnants issus du précédent Benchmark The GridSearch nous a donné :
    # RF : n_estimators=100, max_depth=5, class_weight='balanced'
    # GBM : n_estimators=50, max_depth=3, learning_rate=0.1. (Mais GBM n'a pas class_weight)
    
    # Création des estimateurs de base
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    
    gbm = GradientBoostingClassifier(
        n_estimators=50, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    
    # Voting Classifier "soft" pour moyenner les probabilités
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('gbm', gbm)],
        voting='soft'
    )
    
    # On utilise un GridSearch pour trouver le meilleur équilibre des poids (RF vs GBM)
    # L'objectif d'optimisation est le 'f1', le point d'équilibre exact entre Recall et Precision
    params = {'weights': [[1, 1], [2, 1], [1, 2], [1.5, 1], [1, 1.5]]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(voting_clf, params, cv=cv, scoring='f1', n_jobs=-1)
    
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    best_weights = gs.best_params_['weights']
    logging.info(f"   Meilleurs poids (RF, GBM) pour le F1-Score : {best_weights}")
    
    # PRÉDICTIONS : Le Juste Milieu (Compromis Sécurité / Confort)
    # Seuil ajusté à 38% de probabilité. L'IA trouve son 'sweet spot' entre
    # le seuil paranoïaque (30%) et le seuil confiant (45%).
    y_prob = best_model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else best_model.predict(X_test)
    y_pred = (y_prob >= 0.38).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Feature importances
    rf_imp = best_model.named_estimators_['rf'].feature_importances_
    gbm_imp = best_model.named_estimators_['gbm'].feature_importances_
    # Moyenne pondérée selon les poids validés par le GridSearch
    total_weight = sum(best_weights)
    importances = (rf_imp * best_weights[0] + gbm_imp * best_weights[1]) / total_weight
    
    # --- GRAPHIQUES INDIVIDUELS ---
    sns.set_theme(style="whitegrid")
    
    # Matrice de confusion
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
                xticklabels=['Normal', 'Handover'], yticklabels=['Normal', 'Handover'])
    plt.title('Modèle Hybride (RF+GBM) - Matrice de Confusion')
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    cm_path = results_dir / "hybrid_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Feature Importances (Top 10)
    plt.figure(figsize=(6, 4))
    indices = np.argsort(importances)[::-1][:10]
    sns.barplot(x=importances[indices], y=[features_names[i] for i in indices], palette='plasma')
    plt.title('Hybride - Top 10 Facteurs (Moyenne RF+GBM)')
    fi_path = results_dir / "hybrid_feature_importances.png"
    plt.savefig(fi_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return {
        'name': 'Hybride (RF+GBM)',
        'recall': report['1']['recall'],
        'precision': report['1']['precision'],
        'f1': report['1']['f1-score'],
        'accuracy': report['accuracy'],
        'cm': cm,
        'importances': importances,
        'y_prob': y_prob,
        'cm_path': cm_path.name,
        'fi_path': fi_path.name
    }
