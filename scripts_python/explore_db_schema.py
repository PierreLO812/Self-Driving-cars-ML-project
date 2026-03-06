import sqlite3
import sys
from pathlib import Path

def explore_schema(db_path):
    print(f"==================================================")
    print(f" Exploration du schéma de la DB : {Path(db_path).name}")
    print(f"==================================================\n")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. Récupérer toutes les tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("Aucune table trouvée dans cette base de données.")
            return

        for table in tables:
            table_name = table[0]
            print(f"--- Table: {table_name} ---")
            
            # 2. Pour chaque table, afficher ses colonnes
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns = cursor.fetchall()
            
            # Format PRAGMA table_info : 
            # cid (0), name (1), type (2), notnull (3), dflt_value (4), pk (5)
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                print(f"  - {col_name} ({col_type if col_type else 'TYPE INCONNU'})")
            print()

        conn.close()
    except Exception as e:
        print(f"Erreur lors de l'ouverture ou la lecture de la base : {e}")

if __name__ == "__main__":
    # Chemin racine, on va chercher une base au hasard ou utiliser une en particulier
    # On va chercher la première base .db dans NUPLAN_ROOT_DIR
    target_dir = Path(r"E:\ML\nuplan-v1.0_train_singapore")
    
    # Cherchons le fichier mentionné ou un autre
    dbs = list(target_dir.rglob("*.db"))
    
    if len(dbs) > 0:
        target_db = None
        for db in dbs:
            if "2021.08.18.06.04.33_veh-51_00016_00170.db" in db.name:
                target_db = db
                break
        
        # Si on n'a pas trouvé le spécifique, on prend le premier
        if target_db is None:
            target_db = dbs[0]
            
        explore_schema(target_db)
    else:
        print(f"Aucun fichier .db trouvé dans {target_dir}")
