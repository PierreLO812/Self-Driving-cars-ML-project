import sqlite3
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION NUPLAN ---
NUPLAN_ROOT_DIR = r"E:\ML\nuplan-v1.0_train_singapore\data\cache\public_set_sg_train"
# ----------------------------

def extract_features_from_db(db_path):
    """
    Extrait les variables cinématiques (ego_pose) et la densité du trafic (lidar_box)
    pour construire la matrice X d'une base SQLite nuPlan.
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # 1. Extraction Ego Pose (Comportement de la voiture)
        query_ego = """
            SELECT timestamp, vx, vy, acceleration_x, acceleration_y, angular_rate_z 
            FROM ego_pose
        """
        df_ego = pd.read_sql_query(query_ego, conn)
        
        # 2. Extraction Lidar (Densité de trafic environnant)
        # On passe par la table lidar_pc pour lier les objets au timestamp
        query_lidar = """
            SELECT lp.timestamp, COUNT(lb.token) as num_objects
            FROM lidar_pc lp
            LEFT JOIN lidar_box lb ON lp.token = lb.lidar_pc_token
            GROUP BY lp.timestamp
        """
        df_lidar = pd.read_sql_query(query_lidar, conn)
        
        # 3. Scénarios critiques (Optionnel, aide pour le label)
        query_scenario = """
            SELECT lp.timestamp, st.type as scenario_type
            FROM scenario_tag st
            JOIN lidar_pc lp ON st.lidar_pc_token = lp.token
        """
        df_scenario = pd.read_sql_query(query_scenario, conn)
        
        conn.close()
        
        # Fusion des données sur le timestamp
        # Utilisation de merge 'asof' (sur le timestamp le plus proche) car capteurs asynchrones
        df_ego = df_ego.sort_values('timestamp')
        df_lidar = df_lidar.sort_values('timestamp')
        df_scenario = df_scenario.sort_values('timestamp')
        
        # On fusionne la Lidar density sur l'Ego Pose le plus proche (tolérance 50ms = 50 000 µs)
        df_merged = pd.merge_asof(df_ego, df_lidar, on='timestamp', direction='nearest', tolerance=50000)
        
        # On fusionne les tags de scénarios (S'il n'y en a pas, c'est 'nominal')
        df_merged = pd.merge_asof(df_merged, df_scenario, on='timestamp', direction='nearest', tolerance=50000)
        
        # Nettoyage des NaNs post-fusion
        df_merged['num_objects'] = df_merged['num_objects'].fillna(0)
        df_merged['scenario_type'] = df_merged['scenario_type'].fillna('nominal')
        
        return df_merged
        
    except sqlite3.OperationalError as e:
        logging.warning(f"Erreur SQL sur {db_path.name} : {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Erreur d'extraction sur {db_path.name} : {e}")
        return pd.DataFrame()


def define_handover_target(df):
    """
    Applique l'heuristique métier pour définir la Target binaire 'Handover_Required'.
    Target = 1 SI :
    - Freinage d'urgence (acceleration_x < -1.0)
    - OU Embardée/Lacet brutal (abs(angular_rate_z) > 0.3)
    - OU Scène ultra complexe combinée à une densité (>40 objets ET tag évoquant un piéton/danger)
    Sinon Target = 0.
    """
    if df.empty:
        return df
        
    # Heuristique
    condition_brake = df['acceleration_x'] < -1.0
    condition_swerve = df['angular_rate_z'].abs() > 0.3
    
    # tags critiques (basés sur l'EDA)
    critical_tags = ['near_pedestrian_on_crosswalk', 'near_trafficcone_on_driveable', 'on_intersection']
    condition_complex = (df['num_objects'] > 40) & (df['scenario_type'].isin(critical_tags))
    
    # Création de la target
    df['Target_Handover'] = np.where(condition_brake | condition_swerve | condition_complex, 1, 0)
    
    # On peut "drop" la colonne `scenario_type` car on s'en est servi pour la Target
    # ou la transformer en Dummies (One-Hot) si on veut qu'elle soit une Feature de X.
    # Dans ce script, on la garde en Feature One-Hot
    df = pd.get_dummies(df, columns=['scenario_type'], prefix='scene', dummy_na=False)
    
    return df


def build_global_dataset(max_dbs=5):
    """
    Parcourt un échantillon de bases SQLite pour créer un gros Dataset Pandas X, y.
    """
    root_path = Path(NUPLAN_ROOT_DIR)
    db_files = list(root_path.rglob('*.db'))
    
    if not db_files:
        logging.error(f"Aucune base SQLite trouvée dans {root_path}")
        return pd.DataFrame()
        
    all_dfs = []
    dbs_to_process = db_files[:max_dbs] if max_dbs else db_files
    
    logging.info(f"Début de l'extraction sur {len(dbs_to_process)} bases de données...")
    
    for db_path in dbs_to_process:
        df_db = extract_features_from_db(db_path)
        if not df_db.empty:
            df_db = define_handover_target(df_db)
            all_dfs.append(df_db)
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Remplacer les valeurs booléennes des colonnes One-Hot (issues de pd.get_dummies) en entiers
        for col in final_df.columns:
             if final_df[col].dtype == 'bool':
                 final_df[col] = final_df[col].astype(int)
        
        # Remplir les NaNs potentiels dus au concat de différents scénarios
        final_df = final_df.fillna(0)
        
        target_count = final_df['Target_Handover'].sum()
        logging.info(f"Dataset Globale : {len(final_df)} lignes extraites.")
        logging.info(f"Nombre total de Handovers (Target=1) générés par l'heuristique : {target_count}")
        return final_df
    else:
        return pd.DataFrame()


def train_random_forest(df):
    """
    Entraîne le modèle Random Forest sur le dataset complet.
    """
    if df.empty or 'Target_Handover' not in df.columns:
        logging.error("Dataset vide ou Target manquant, annulation de l'entraînement.")
        return
        
    logging.info("Préparation du Machine Learning...")
    
    # On retire le timestamp qui n'est pas prédictif, la target,
    # AINSI QUE les variables cinématiques pour que le Random Forest 
    # ne triche pas et se base uniquement sur l'environnement.
    cols_to_drop = ['timestamp', 'Target_Handover', 'vx', 'vy', 'acceleration_x', 'acceleration_y', 'angular_rate_z']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['Target_Handover']
    
    # Séparation 80/20 Chronologique préférée en conduite autonome (ou random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y if y.sum()>1 else None)
    
    # RobustScaler pour gérer les outliers cinématiques (ex: accélération extrême due au bruit)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logging.info("Entraînement du Random Forest (class_weight='balanced' pour la rareté du Handover)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    
    clf.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = clf.predict(X_test_scaled)
    
    # Métriques
    print("\n" + "="*50)
    print(" RÉSULTATS DU MODÈLE RANDOM FOREST (HANDOVER) ")
    print("="*50)
    print("Matrice de Confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport de Classification :")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("="*50)
    
    # Feature Importances
    importances = clf.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\nTop 5 Variables les plus importantes :")
    for i in range(min(5, len(feature_names))):
        print(f"{i+1}. {feature_names[sorted_idx[i]]} ({importances[sorted_idx[i]]:.4f})")
    print("="*50 + "\n")


if __name__ == "__main__":
    # Etape 1 : Construction du Dataset (Limité à 10 DB pour commencer/tester)
    # Remplacer max_dbs par None pour tout traiter
    final_dataset = build_global_dataset(max_dbs=10)
    
    # Etape 2 : Entraînement et Evaluation
    train_random_forest(final_dataset)
