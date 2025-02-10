# Projet SMART (Smart Merchandise Automated Recognition Technology)

## Description 📌
Le projet SMART vise à développer une solution de Computer Vision en Python capable de reconnaître automatiquement un ensemble défini de 10 produits. Le projet applique les bonnes pratiques de développement et les concepts MLOps vus en cours.

## Contraintes ⚙️
- Projet réalisé en groupe de 2 personnes.
- Utilisation d'un GPU NVIDIA ou d'une puce Apple M1+ (ou des machines de TPS).
- Versionné sur GitHub.
- Langage : Python 3.11.
- Gestion des datasets et annotation : **Picsellia**.
- Entraînement des modèles : **YOLO d'Ultralytics**.
- Experiment Tracking et Model Registry : **MLFlow**.
- Déploiement des modèles : **BentoML**.
- Monitoring des modèles déployés **non requis**.

## Produits reconnus 🎯
- Granola
- Balisto_violet
- Thon
- Bouteille_plastique
- Bueno_white
- Bueno_black
- Tablette_chocolat
- Kinder_Délice
- Snickers
- Twix

---

## Installation 🚀

### Prérequis
- Python 3.11
- GPU NVIDIA / Apple M1+
- Bibliothèques requises :
```sh
pip install -r requirements.txt
```

### Configuration du projet
1. **Cloner le dépôt** :
```sh
git clone https://github.com/EnzoBergamini/SMART-PROJECT.git
cd SMART-PROJECT
```

2. **Configuration du venv**
```sh
uv venv --python 3.11 .venv

uv pip install -r requirements.txt
```

3. **Configuration de pre-commit**
```sh
pre-commit install
```

---

## Pipeline de Training 🏋️‍♂️
1. **Récupérer le dataset annoté** :
   - Utiliser le SDK de **Picsellia**.
   - Configurer l'ID du dataset via `argparse` ou `config.py`.
2. **Pre-processing** :
   - Split des données (train : 60%, val : 20%, test : 20%).
   - Génération automatique du fichier `config.yaml`.
3. **Training du modèle YOLO** :
   - Architecture YOLOv11 (nano, small, medium, large ou XL).
   - Log des métriques (loss, accuracy) sur **MLFlow**.
4. **Stockage du modèle dans le Model Registry** :
   - Stockage du `best.pt` avec **MLFlow**.
   - Tagging du modèle en "Champion" ou "Challenger".
5. **Déploiement avec BentoML** :
   - Déploiement dans le cloud avec **BentoML**.
   - Test des inférences post-déploiement.

---

## Pipeline d’Inférence 🔍
1. **Récupération du modèle "Champion"** :
   - Téléchargement depuis **MLFlow**.
2. **Inférence** :
   - Mode configurable : `IMAGE`, `VIDEO` ou `WEBCAM`.
   - Utilisation de **YOLO** pour la détection.

---

## Liens utiles 🔗
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [MLFlow Docs](https://mlflow.org/docs/latest/index.html)
- [BentoML Docs](https://www.bentoml.com/)
- [Picsellia](https://www.picsellia.com/)
