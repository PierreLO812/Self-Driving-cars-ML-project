import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

def run_model(X_train, X_test, y_train, y_test, features_names, results_dir):
    logging.info("--- Entraînement Linear SVM ---")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    model = LinearSVC(class_weight='balanced', random_state=42, dual=False)
    params = {'C': [0.1, 1.0]}
    
    gs = GridSearchCV(model, params, cv=cv, scoring='recall', n_jobs=-1)
    gs.fit(X_train, y_train)
    
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # LinearSVC n'a pas predict_proba par défaut
    decision = best_model.decision_function(X_test)
    y_prob = (decision - decision.min()) / (decision.max() - decision.min())
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    importances = np.abs(best_model.coef_[0])
    
    # --- GRAPHIQUES INDIVIDUELS ---
    sns.set_theme(style="whitegrid")
    
    # Matrice de confusion
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', cbar=False,
                xticklabels=['Normal', 'Handover'], yticklabels=['Normal', 'Handover'])
    plt.title('Linear SVM - Matrice de Confusion')
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    cm_path = results_dir / "svm_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Feature Importances (Top 10)
    plt.figure(figsize=(6, 4))
    indices = np.argsort(importances)[::-1][:10]
    sns.barplot(x=importances[indices], y=[features_names[i] for i in indices], palette='bone')
    plt.title('SVM - Top 10 Coefficients Absolus')
    fi_path = results_dir / "svm_feature_importances.png"
    plt.savefig(fi_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return {
        'name': 'Linear SVM',
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
