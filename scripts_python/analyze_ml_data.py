import sqlite3
import pandas as pd
from pathlib import Path

def explore_ml_features(db_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"==================================================\n")
        f.write(f" Analyse des données pour ML : {db_path.name}\n")
        f.write(f"==================================================\n\n")
        try:
            conn = sqlite3.connect(db_path)

            # 1. Vérifions les types de scénarios (Tags) pour voir si un vrai label existe
            f.write("--- SCENARIO TAGS (Valeurs Uniques) ---\n")
            try:
                tags = pd.read_sql_query("SELECT DISTINCT type FROM scenario_tag;", conn)
                if not tags.empty:
                    f.write(tags.to_string() + "\n")
                else:
                    f.write("Aucun tag de scénario trouvé.\n")
            except Exception as e:
                f.write(f"Erreur lecture scenario_tag: {e}\n")

            # 2. Statistiques Cinématiques du véhicule Ego (Ego Pose)
            f.write("\n--- EGO POSE (Cinématique du véhicule - 5 premières lignes) ---\n")
            try:
                df_ego = pd.read_sql_query("SELECT timestamp, vx, vy, acceleration_x, acceleration_y, angular_rate_z FROM ego_pose LIMIT 5;", conn)
                f.write(df_ego.to_string() + "\n")
                
                # Afficher des stats globales sur l'accélération pour déterminer un seuil de "freinage d'urgence" (Handover proxy)
                df_ego_stats = pd.read_sql_query("SELECT min(acceleration_x) as min_accel, max(acceleration_x) as max_accel, avg(acceleration_x) as avg_accel FROM ego_pose;", conn)
                f.write(f"\nStats Accélération (X) : {df_ego_stats.to_dict(orient='records')}\n")
            except Exception as e:
                f.write(f"Erreur lecture ego_pose: {e}\n")

            # 3. Compter le nombre de Lidar Boxes par frame Lidar (Densité de trafic)
            f.write("\n--- LIDAR BOXES (Aperçu du trafic autour de la voiture) ---\n")
            try:
                df_boxes = pd.read_sql_query("""
                    SELECT lidar_pc_token, COUNT(token) as num_objects 
                    FROM lidar_box 
                    GROUP BY lidar_pc_token 
                    LIMIT 20;
                """, conn)
                f.write("Nombre d'objets détectés par numérisation lidar (20 frames) :\n")
                f.write(df_boxes.to_string() + "\n")
            except Exception as e:
                f.write(f"Erreur lecture lidar_box: {e}\n")

            conn.close()
        except Exception as e:
            f.write(f"Erreur globale : {e}\n")

if __name__ == "__main__":
    target_dir = Path(r"E:\ML\nuplan-v1.0_train_singapore\data\cache\public_set_sg_train")
    dbs = list(target_dir.rglob("*.db"))
    
    output_path = target_dir.parent / "ml_EDA_report.md"
    
    if len(dbs) > 0:
        target_db = dbs[0]
        explore_ml_features(target_db, output_path)
        print(f"Rapport généré dans {output_path}")
    else:
        print(f"Aucun fichier .db trouvé dans {target_dir}")
