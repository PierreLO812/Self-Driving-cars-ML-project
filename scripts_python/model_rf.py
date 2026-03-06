import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def run_model(X_train, X_test, y_train, y_test, features_names, results_dir):
    logging.info("--- Entraînement Random Forest ---")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    
    gs = GridSearchCV(model, params, cv=cv, scoring='recall', n_jobs=-1)
    gs.fit(X_train, y_train)
    
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else y_pred
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    importances = best_model.feature_importances_
    
    # --- GRAPHIQUES INDIVIDUELS ---
    sns.set_theme(style="whitegrid")
    
    # Matrice de confusion
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Handover'], yticklabels=['Normal', 'Handover'])
    plt.title('Random Forest - Matrice de Confusion')
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    cm_path = results_dir / "rf_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Feature Importances (Top 10)
    plt.figure(figsize=(6, 4))
    indices = np.argsort(importances)[::-1][:10]
    sns.barplot(x=importances[indices], y=[features_names[i] for i in indices], palette='viridis')
    plt.title('RF - Top 10 Facteurs de Désengagement')
    fi_path = results_dir / "rf_feature_importances.png"
    plt.savefig(fi_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return {
        'name': 'Random Forest',
        'recall': report['1']['recall'],
        'precision': report['1']['precision'],
        'f1': report['1']['f1-score'],
        'accuracy': report['accuracy'],
        'cm': cm,
        'importances': importances,
        'best_params': gs.best_params_,
        'y_prob': y_prob,
        'cm_path': cm_path.name,
        'fi_path': fi_path.name
    }
