# Prédiction à 2 Frames à l'Avance (Anticipation Handover)

Le modèle actuel calcule la target `Target_Handover` sur la frame **courante** (t) en regardant si la frame *t* elle-même subit un freinage, une embardée ou une situation complexe.

L'objectif est de **décaler la target de +2 frames** : à la frame `t`, le modèle doit prédire si la frame `t+2` sera critique — permettant à la voiture d'anticiper et de réagir à l'avance.

## Proposed Changes

### 03_tabular_feature_model.py

#### [MODIFY] [03_tabular_feature_model.py](file:///d:/SelfDrivingCars/scripts_python/03_tabular_feature_model.py)

La fonction [define_handover_target()](file:///d:/SelfDrivingCars/scripts_python/03_tabular_feature_model.py#78-107) crée actuellement `Target_Handover` sur la frame t. On va ajouter une étape de **shift temporel** : après création de la target brute, on la décale de -2 positions (`.shift(-2)`) pour que chaque ligne `t` contienne le label de la ligne `t+2`. Les 2 dernières lignes (sans futur) sont supprimées.

**Changement clé :**
```python
# AVANT
df['Target_Handover'] = np.where(condition_brake | condition_swerve | condition_complex, 1, 0)

# APRÈS
target_raw = np.where(condition_brake | condition_swerve | condition_complex, 1, 0)
df['Target_Handover'] = pd.Series(target_raw, index=df.index).shift(-2)
df = df.dropna(subset=['Target_Handover'])
df['Target_Handover'] = df['Target_Handover'].astype(int)
```

> [!IMPORTANT]
> Le shift doit être fait **avant** le `pd.get_dummies`, donc on réorganise légèrement l'ordre dans [define_handover_target](file:///d:/SelfDrivingCars/scripts_python/03_tabular_feature_model.py#78-107). Il faut aussi que les frames soient **dans l'ordre chronologique** (trié par `timestamp`) avant le shift — ce qui est déjà le cas grâce au `sort_values('timestamp')` dans [extract_features_from_db](file:///d:/SelfDrivingCars/scripts_python/03_tabular_feature_model.py#17-76).

#### Mise à jour du commentaire docstring de [define_handover_target](file:///d:/SelfDrivingCars/scripts_python/03_tabular_feature_model.py#78-107)
Ajout de la mention **"Prédiction anticipatoire à +2 frames"** dans la docstring pour traçabilité.

---

### main.py & run_hybrid.py

#### [MODIFY] [main.py](file:///d:/SelfDrivingCars/scripts_python/main.py)
#### [MODIFY] [run_hybrid.py](file:///d:/SelfDrivingCars/scripts_python/run_hybrid.py)

Mise à jour des rapports Markdown générés pour mentionner explicitement **"Prédiction à +2 frames"** dans les titres et descriptions, pour que les résultats soient bien documentés et non confondus avec l'ancienne approche.

---

## Verification Plan

### Test par script de vérification rapide

Créer `/tmp/verify_shift.py` — script de test sans SQLite (données synthétiques) qui :
1. Construit un mini-DataFrame avec des patterns connus (events à t=5 et t=10)
2. Appelle [define_handover_target()](file:///d:/SelfDrivingCars/scripts_python/03_tabular_feature_model.py#78-107)
3. Vérifie que `Target_Handover=1` se trouve bien aux lignes **t=3 et t=8** (soit 2 frames avant l'event)

**Commande :**
```powershell
python /tmp/verify_shift.py
```
