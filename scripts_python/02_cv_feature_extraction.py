"""
Pipeline Computer Vision - Feature Engineering pour SafeHandover-ML.

Ce script traite un dossier d'images brutes de caméras (nuPlan/nuScenes)
pour en extraire des variables contextuelles utiles au modèle de ML.

Il utilise YOLOv8 pour la détection d'objets (véhicules, VRU) et 
des heuristiques/simulations pour l'état de la route et de l'atmosphère.
Le résultat est exporté sous forme d'un fichier CSV fusionnable sur le 'timestamp'.
"""

import os
import glob
import logging
import cv2
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from pathlib import Path

# --- CONFIGURATION DU DOSSIER NUPLAN ---
# Modifie ce chemin pour pointer vers le dossier racine contenant tes données nuPlan
# (Il peut pointer vers un dossier parent qui contient de nombreux sous-dossiers d'images)
NUPLAN_ROOT_DIR = r"E:\ML\nuplan-v1.0_train_singapore\data\cache\public_set_sg_train" # À AJUSTER SELON TON PC
# ---------------------------------------

try:
    from ultralytics import YOLO
except ImportError:
    logging.warning("Ultralytics n'est pas installé. Lancez : pip install ultralytics")
    YOLO = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CVFeatureExtractor:
    def __init__(self, model_version='yolov8n.pt'):
        """
        Initialise le pipeline CV.
        Charge YOLOv8 pour traquer le trafic dynamique.
        """
        logging.info(f"Chargement du modèle de détection d'objets : {model_version}")
        if YOLO is not None:
            try:
                # Modeleur pré-entraîné sur dataset COCO
                self.detector = YOLO(model_version)
            except Exception as e:
                logging.error(f"Erreur lors du chargement de YOLOv8 : {e}")
                self.detector = None
        else:
            self.detector = None

        # Classes COCO pour le tri
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        # 9: traffic light, 28: suitcase/umbrella (Souvent utilisé comme proxy pour traffic cones dans COCO brut)
        self.vehicle_classes = [2, 3, 5, 7]
        self.vru_classes = [0, 1]

    def _estimate_distance(self, bbox, frame_width, frame_height):
        """
        1. Les Agents Dynamiques : Distance
        Estime la distance d'un objet en fonction de la taille de sa bounding box.
        Heuristique (pinhole camera). Retourne une distance en mètres.
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        
        # Dimensions standard pour une voiture: 1.5m de haut.
        # Focale approximative (à ajuster selon la caméra nuScenes)
        focal_length = 800  # focal in pixels
        real_height = 1.5   # mètres
        
        if bbox_height > 0:
            distance = (real_height * focal_length) / bbox_height
            return min(max(distance, 0), 100) # Borne [0, 100m]
        return 100.0

    def analyze_image(self, img_array, timestamp, log_id="unknown"):
        """
        Analyse de l'image (décodée en numpy array) pour extraire les 9 features demandées.
        """
        features = {
            'timestamp': timestamp,
            'log_id': log_id,
            # 1. Agents Dynamiques
            'vehicle_count': 0,
            'pedestrian_and_cyclist_count': 0,
            'closest_object_distance': 100.0,
            # 2. Infrastructure et Situation
            'intersection_complexity': 0,
            'construction_zone': 0,
            'lane_markings_visibility': 0.8,
            # 3. Atmosphérique / Capteur
            'weather_severity_score': 0,
            'illumination_level': 'Day',
            'camera_obstruction_or_glare': 0
        }

        # Si YOLO n'est pas dispo, on gère l'image avec des modèles spécialisés basiques.
        try:
            if img_array is None or img_array.size == 0:
                logging.error(f"Image invalide pour le timestamp {timestamp}")
                return features

            frame_height, frame_width = img_array.shape[:2]

            # A. Inférence YOLOv8 (Objets dynamiques)
            if self.detector is not None:
                # results est une liste (un résultat par frame, batch de 1)
                results = self.detector(img_array, verbose=False)[0]
                boxes = results.boxes
                
                closest_dist = 100.0
                traffic_light_count = 0
                
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    if cls_id in self.vehicle_classes:
                        features['vehicle_count'] += 1
                    elif cls_id in self.vru_classes:
                        features['pedestrian_and_cyclist_count'] += 1
                    elif cls_id == 9: # Traffic Light
                        traffic_light_count += 1
                    
                    # Cacul distance du plus proche (n'importe quel type d'obstacle)
                    dist = self._estimate_distance(xyxy, frame_width, frame_height)
                    if dist < closest_dist:
                        closest_dist = dist
                        
                features['closest_object_distance'] = round(closest_dist, 2)
                
                # Heuristique Intersection (Très Dense = >=10 véhicules ET >= 1 Feu Rouge)
                if features['vehicle_count'] >= 10 and traffic_light_count > 0:
                    features['intersection_complexity'] = 1
                    
            # B. Inférence CNN spécialisés / OpenCV
            # Météo (0=Clair, 5=Extrême)
            features['weather_severity_score'] = self._simulate_weather_classifier(img_array)
            
            # Illumination
            features['illumination_level'] = self._classify_illumination(img_array)
            
            # Glare / Obstruction
            features['camera_obstruction_or_glare'] = self._detect_camera_glare(img_array)
            
            # Marquage au sol (Lignes Blanches/Jaunes)
            features['lane_markings_visibility'] = self._estimate_lane_visibility(img_array)
            
            # Zone de travaux (Heuristique ou Placeholder CNN)
            features['construction_zone'] = np.random.choice([0, 1], p=[0.9, 0.1])

        except Exception as e:
            logging.error(f"Erreur inattendue sur {timestamp} : {e}")

        return features

    # --- Sous-modèles d'analyse de Scène (Simulés ou heuristiques OpenCV) ---
    
    def _classify_illumination(self, img):
        """ Heuristique simple: la luminosité moyenne en HSV pour déterminer Jour/Nuit/Crépuscule """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = hsv[..., 2].mean()
        
        if brightness < 60:
            return 'Night'
        elif brightness < 110:
            return 'Twilight'
        return 'Day'

    def _detect_camera_glare(self, img):
        """ Détecte l'aveuglement caméra (Lens glare) en comptant les zones brûlées """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        overexposed = np.sum(gray > 240) / gray.size # Pixels très proches du blanc
        
        # Si > 5% de la caméra est brûlée
        return 1 if overexposed > 0.05 else 0

    def _simulate_weather_classifier(self, img):
        """ 
        En prod : Modèle ResNet50. 
        Ici heuristique (si très faible constraste -> brouillard/pluie forte).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        
        if contrast < 20: # Brouillard épais par exemple
            return np.random.randint(3, 6)
        if contrast < 40:
            return np.random.randint(1, 3)
        return 0

    def _estimate_lane_visibility(self, img):
        """ Extrait les arêtes à forte inclinaison (lignes de route) pour estimer la visibilité """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Ratio de surface occupée par des "contours" (très simplifié)
        ratio = np.sum(edges > 0) / edges.size
        # Normalisation heuristique entre [0, 1]
        score = min(max(ratio * 50, 0), 1)
        return round(score, 2)


def discover_image_base_path(root_dir, target_filename_jpg):
    """
    Rétro-ingénierie : cherche physiquement le fichier target_filename_jpg dans root_dir.
    Une fois trouvé, il déduit le "base_path" parent pour éviter de refaire la recherche.
    Exemple:
      target_filename_jpg = "cam_f0/image_123.jpg"
      trouvé à = "E:/ML/nuplan/sensor_data/log_ABC/cam_f0/image_123.jpg"
      retourne = "E:/ML/nuplan/sensor_data/log_ABC"
    """
    target_path_obj = Path(target_filename_jpg)
    target_name = target_path_obj.name # juste "image_123.jpg"
    
    logging.info(f"Détective : Recherche de la 1ère image '{target_name}' depuis {root_dir}")
    
    # Recherche du fichier spécifique (beaucoup plus rapide qu'un rglob de '*.jpg')
    for found_path in Path(root_dir).rglob(target_name):
        # On a trouvé le fichier physique !
        # On vérifie si la fin du chemin correspond bien au filename_jpg de la DB.
        # Ex: vérifier si "cam_f0/image_123.jpg" est bien à la fin de "E:/.../cam_f0/image_123.jpg"
        found_str = str(found_path).replace('\\', '/')
        target_str = str(target_filename_jpg).replace('\\', '/')
        
        if found_str.endswith(target_str):
            # Le chemin de base est trouvé physiquement en soustrayant target_str
            base_str = found_str[:-len(target_str)].rstrip('/')
            logging.info(f"Détective : Pattern trouvé ! Base path = {base_str}")
            return Path(base_str)
            
    logging.warning(f"Détective : Impossible de trouver '{target_filename_jpg}' sur le disque.")
    return None

def process_db_file(db_path, extractor, metrics_list, max_images, root_dir):
    """
    Se connecte à un fichier .db SQLite nuPlan, extrait les chemins (filename_jpg) et lit les vraies images.
    """
    logging.info(f"Connexion à la base de données : {db_path.name}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Le schéma NuPlan montre que les détails de l'image (filename_jpg) sont dans 'image',
        # et que le 'channel' (ex: CAM_F0) est dans 'camera'.
        # On utilise le 'camera_token' pour faire la jointure.
        query = """
            SELECT i.timestamp, i.filename_jpg 
            FROM image i
            JOIN camera c ON i.camera_token = c.token
            WHERE c.channel = 'CAM_F0'
        """ 
        
        # Essai de la requête
        try:
             cursor.execute(query)
        except sqlite3.OperationalError as e:
             logging.warning(f"La requête SQL a échoué dans {db_path.name} : {e}")
             conn.close()
             return len(metrics_list)
             
        # Le log_id peut simplement être le nom de la base de données courante
        log_id = db_path.stem 
        
        # Ce pattern ("base_path") sera déduit à la volée une seule fois par log (DB)
        base_path = None
        
        while True:
            # Récupérer quelques lignes à la fois pour ne pas exploser la RAM
            rows = cursor.fetchmany(10)
            if not rows:
                 break
                 
            for row in rows:
                if max_images is not None and len(metrics_list) >= max_images:
                    conn.close()
                    return len(metrics_list)
                    
                timestamp = row[0]
                filename_jpg = row[1]
                
                # --- ÉTAPE DE DÉTECTIVE (1 seule fois par DB) ---
                if base_path is None:
                    base_path = discover_image_base_path(root_dir, filename_jpg)
                    if base_path is None:
                        logging.error("L'auto-découverte a échoué. Arrêt du traitement pour cette DB.")
                        conn.close()
                        return len(metrics_list)
                
                # Génération instantanée du chemin avec le pattern mémorisé
                image_full_path = base_path / filename_jpg
                
                if not image_full_path.exists():
                     logging.warning(f"Image manquante : {image_full_path}")
                     continue
                
                # Chargement pur de l'image
                img = cv2.imread(str(image_full_path))
                
                if img is not None:
                    features = extractor.analyze_image(img, timestamp, log_id)
                    metrics_list.append(features)
                    
                    if len(metrics_list) % 10 == 0:
                        logging.info(f"-> {len(metrics_list)} frames traitées globalement.")
                else:
                    logging.warning(f"Impossible de lire l'image {timestamp} au chemin : {image_full_path}")
                    
        conn.close()
    except Exception as e:
        logging.error(f"Erreur lors du traitement de {db_path.name} : {e}")

    return len(metrics_list)


def generate_pipeline_from_db(db_root_dir, output_csv, max_images=None):
    """
    Recherche récursivement les fichiers .db dans db_root_dir, lit les chemins d'images cibles 
    sur le disque et construit le dataframe de Preprocessing CV.
    """
    extractor = CVFeatureExtractor(model_version='yolov8n.pt')
    
    root_path = Path(db_root_dir)
    logging.info(f"Recherche récursive de bases SQLite (.db) dans : {root_path}")
    
    # Recherche récursive
    db_files = list(root_path.rglob('*.db'))
    
    if len(db_files) == 0:
        logging.warning("Aucune base .db trouvée dans le répertoire.")
        return
        
    logging.info(f"Nombre de fichiers .db trouvés : {len(db_files)}")
    
    metrics = []
    
    for db_path in db_files:
        if max_images is not None and len(metrics) >= max_images:
            logging.info(f"Limite de {max_images} images atteinte.")
            break
            
        process_db_file(db_path, extractor, metrics, max_images, root_path)

    if len(metrics) > 0:
        df_vision = pd.DataFrame(metrics)
        # Optionnel: on peut aggréger ou cleaner ici
        # df_vision['vehicle_count'] = df_vision['vehicle_count'].clip(upper=50) 
        
        df_vision.to_csv(output_csv, index=False)
        logging.info(f"Pipeline terminé avec succès. Les features CV ont été exportées dans : {output_csv}")
    else:
        logging.warning("Aucune métrique n'a pu être extraite des DB.")

if __name__ == "__main__":
    # Paramétrages de répertoire de sortie
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    CSV_OUTPUT_PATH = os.path.join(DATA_DIR, "vision_features.csv")
    
    # ---------------- TEST MODE ---------------- 
    # Mettre MAX_IMAGES_TO_PROCESS = None pour tout traiter
    MAX_IMAGES_TO_PROCESS = 100 
    # -------------------------------------------
    
    # Lancement du Feature Engineering Vision sur les bases SQLite ()
    generate_pipeline_from_db(NUPLAN_ROOT_DIR, CSV_OUTPUT_PATH, max_images=MAX_IMAGES_TO_PROCESS)
