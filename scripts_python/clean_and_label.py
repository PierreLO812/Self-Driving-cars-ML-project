"""
Script de nettoyage et labellisation pour le projet SafeHandover-ML.

Ce script est conçu pour traiter les datasets massifs (nuPlan/nuScenes)
en utilisant des générateurs (chunks) pour ne jamais charger la 
totalité des données en mémoire en une fois.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(df):
    """
    Nettoyage des données :
    - Traitement des valeurs manquantes.
    - Suppression des outliers (basé sur le critère du professeur / statistiques simples).
    """
    # Traitement des valeurs manquantes: on remplit par la médiane pour les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Traitement des outliers (Critère Prof) :
    # Exemple d'implémentation par Z-score modifié ou Intervalle Interquartile (IQR).
    # Ici, nous utilisons l'IQR pour exclure les valeurs aberrantes.
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # On filtre les lignes conservant les valeurs dans les limites
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def create_target(df, confidence_threshold=0.8):
    """
    Création de la colonne Target :
    1 -> si les conditions environnementales (pluie, nuit, trafic dense) 
         dépassent un seuil de confiance (ex: 0.8),
    0 -> sinon.
    """
    # Noms de colonnes hypothétiques issus des annotations nuScenes/nuPlan
    conditions_cols = ['rain_confidence', 'night_confidence', 'heavy_traffic_confidence']
    
    # Sécurité pour l'exemple : si les colonnes n'existent pas, on les simule (utile au début du projet)
    for col in conditions_cols:
        if col not in df.columns:
            # logging.warning(f"Colonne {col} manquante, création factice.")
            df[col] = np.random.uniform(0, 1, size=len(df))
            
    # La condition est remplie si au moins un des facteurs environnementaux est supérieur au seuil
    complex_conditions = (df['rain_confidence'] > confidence_threshold) | \
                         (df['night_confidence'] > confidence_threshold) | \
                         (df['heavy_traffic_confidence'] > confidence_threshold)
                         
    df['Target'] = complex_conditions.astype(int)
    
    return df

def process_large_dataset(file_path, output_path, chunksize=50000):
    """
    Charge et traite les données par lots (chunks) pour éviter la saturation de la RAM.
    """
    logging.info(f"Début du traitement de {file_path} par lots de {chunksize} lignes.")
    
    first_chunk = True
    rows_processed = 0
    
    try:
        # Utilisation de pandas par chunks pour ne pas tout charger en mémoire
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # 1. Nettoyage
            chunk_cleaned = clean_data(chunk)
            
            # 2. Labellisation
            chunk_labeled = create_target(chunk_cleaned)
            
            # 3. Sauvegarde (append après le premier fichier)
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            chunk_labeled.to_csv(output_path, mode=mode, header=header, index=False)
            
            rows_processed += len(chunk)
            logging.info(f"{rows_processed} lignes traitées...")
            first_chunk = False
            
        logging.info(f"Traitement terminé. {rows_processed} lignes sauvegardées dans {output_path}")
    except FileNotFoundError:
        logging.error(f"Erreur : Le fichier {file_path} est introuvable.")
    except Exception as e:
        logging.error(f"Une erreur s'est produite lors du traitement : {str(e)}")

def main():
    # Définition des chemins relatifs aux données
    # Supposons que "data" se trouve à la racine du projet, un niveau au-dessus de scripts_python
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    input_file = os.path.join(data_dir, "mini_split_raw.csv")
    output_file = os.path.join(data_dir, "mini_split_processed.csv")
    
    # Génération d'un mini dataset factice pour l'exemple s'il n'existe pas
    if not os.path.exists(input_file):
        logging.info("Génération d'un jeu de données factice 'Mini Split' pour tester l'architecture...")
        dummy_data = pd.DataFrame({
            'scenario_id': range(200),
            'vehicle_speed': np.random.normal(30, 15, 200),
            'rain_confidence': np.random.uniform(0, 1, 200),
            'night_confidence': np.random.uniform(0, 1, 200),
            'heavy_traffic_confidence': np.random.uniform(0, 1, 200)
        })
        # Injection volontaire de NaNs et Outliers pour vérifier le nettoyage
        dummy_data.loc[5:10, 'vehicle_speed'] = np.nan
        dummy_data.loc[15:20, 'vehicle_speed'] = 200.0  # Outliers
        
        dummy_data.to_csv(input_file, index=False)
    
    # Lancement du traitement
    process_large_dataset(input_file, output_file, chunksize=50)

if __name__ == "__main__":
    main()
