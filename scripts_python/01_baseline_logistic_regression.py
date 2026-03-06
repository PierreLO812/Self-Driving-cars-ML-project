import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mock_nuplan_extraction():
    """
    Simule l'extraction d'un Mini Split depuis nuPlan/nuScenes (via nuplan-devkit).
    Génère un dataset factice avec des valeurs manquantes et des outliers pour tester le pipeline.
    """
    logging.info("Extraction de l'échantillon de données (simulation nuplan-devkit)...")
    np.random.seed(42)
    n_samples = 1000
    
    # Feature Engineering (Variables clés)
    # 0 = Nuit, 1 = Jour pour lighting_condition
    df = pd.DataFrame({
        'visibility': np.random.uniform(0, 100, n_samples),
        'precipitation': np.random.uniform(0, 100, n_samples),
        'traffic_density': np.random.randint(0, 50, n_samples),
        'lighting_condition': np.random.choice([0, 1], n_samples)
    })
    
    # Injection intentionnelle de valeurs manquantes (NaNs)
    for col in ['visibility', 'precipitation', 'traffic_density']:
        mask = np.random.rand(n_samples) < 0.05
        df.loc[mask, col] = np.nan
        
    # Injection intentionnelle d'outliers
    df.loc[10:15, 'visibility'] = 200.0
    df.loc[50:55, 'traffic_density'] = 500
    
    return df

def engeneer_target(df):
    """
    Création de la Target (Label) : 
    Si visibility < 40% OU precipitation > 70%, alors Target = 1 (Handover), sinon 0.
    """
    logging.info("Création de la variable Target...")
    # np.where traite intelligemment les conditions. Les NaN retournent False par défaut dans ces comparateurs.
    df['Target'] = np.where((df['visibility'] < 40) | (df['precipitation'] > 70), 1, 0)
    return df

def clean_data(df):
    """
    Nettoyage du dataset :
    1. Valeurs manquantes : Imputation par la médiane.
    2. Outliers : Application de limites (Clipping) via l'intervalle interquartile (IQR).
    """
    logging.info("Nettoyage des données (Médiane et Outliers)...")
    
    # Séparation Target / Features pour ne pas altérer la Target
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    # 1. Imputation par la médiane
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # 2. Gestion des outliers (IQR)
    for col in X_imputed.columns:
        if col != 'lighting_condition': # Pas d'outliers sur une variable binaire
            Q1 = X_imputed[col].quantile(0.25)
            Q3 = X_imputed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Au lieu de supprimer les lignes (ce qui supprimerait des Targets associées),
            # on plafonne (clip) les outliers à ces valeurs minimales/maximales.
            X_imputed[col] = np.clip(X_imputed[col], lower_bound, upper_bound)
            
    df_clean = pd.concat([X_imputed, y], axis=1)
    return df_clean

def train_baseline(df):
    """
    Implémente une Régression Logistique simple (modèle de référence).
    Validation par train_test_split (80/20) et affichage des métriques.
    """
    logging.info("Entraînement de la Baseline (Régression Logistique)...")
    
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    # Split 80/20 avec graine fixe pour la reproductibilité
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(" RÉSULTATS DE LA BASELINE - RÉGRESSION LOGISTIQUE")
    print("="*50)
    print("Matrice de Confusion :")
    print(conf_matrix)
    print(f"\nF1-Score : {f1:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Pipeline complet d'exécution
    raw_data = mock_nuplan_extraction()
    data_with_target = engeneer_target(raw_data)
    clean_data_df = clean_data(data_with_target)
    
    train_baseline(clean_data_df)
