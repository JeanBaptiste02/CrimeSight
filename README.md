# CrimeSight - Analyse et Prédiction de Crimes

CrimeSight est une application web interactive permettant d'analyser et de prédire les catégories de crimes basée sur un ensemble de données criminelles. L'application offre une interface moderne avec des visualisations interactives et des fonctionnalités de prédiction.

<div align="center">
  <img src="webapp_demo1.png" alt="Interface de l'application" width="800"/>
</div>

## À propos du Projet

CrimeSight permet aux utilisateurs de :
- Visualiser la distribution des crimes par catégorie
- Analyser les tendances par type de rapport
- Explorer les caractéristiques démographiques des auteurs et des victimes
- Prédire la catégorie de crime basée sur des caractéristiques spécifiques

### Dataset

Le projet utilise un dataset de crimes incluant les informations suivantes :
- Caractéristiques des auteurs (âge, genre, race)
- Caractéristiques des victimes (âge, genre, race)
- Type de rapport
- Catégorie de crime
- Disposition du cas

## Installation

1. Clonez le repository :
```bash
git clone https://github.com/votre-username/CrimeSight.git
cd CrimeSight
```

2. Créez un environnement virtuel Python (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Dépendances

Le projet utilise les bibliothèques Python suivantes :
- `flask==3.0.0` : Framework web
- `pandas==2.2.0` : Manipulation et analyse de données
- `numpy==1.26.4` : Calculs numériques
- `scikit-learn==1.4.0` : Machine Learning
- `matplotlib==3.8.2` : Visualisation de données
- `seaborn==0.13.2` : Visualisation statistique
- `joblib==1.3.2` : Sauvegarde des modèles

## Lancement de l'Application

1. Assurez-vous que toutes les dépendances sont installées
2. Lancez l'application Flask :
```bash
python app.py
```
3. Accédez à l'application dans votre navigateur : `http://localhost:5000`