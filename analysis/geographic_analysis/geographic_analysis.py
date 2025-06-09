import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import os

# Configuration de l'affichage
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# Configuration de la taille des figures
plt.rcParams['figure.figsize'] = [12, 6]

def create_img_dir():
    """Crée le dossier img s'il n'existe pas"""
    img_dir = os.path.join(os.path.dirname(__file__), 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    return img_dir

def load_data():
    """Charge les données nettoyées"""
    print("Chargement des données...")
    df = pd.read_csv('../../cleaned_crime_data.csv')
    print(f"Nombre total d'enregistrements : {len(df)}")
    print(df.columns)
    return df

def analyze_regional_distribution(df, img_dir):
    """Analyse la distribution des crimes par type de rapport (Report Type)"""
    print("\nAnalyse de la distribution des crimes par type de rapport...")
    
    # Nombre de crimes par type de rapport
    region_counts = df['Report Type'].value_counts()
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    region_counts.plot(kind='bar')
    plt.title('Distribution des crimes par type de rapport')
    plt.xlabel('Type de rapport')
    plt.ylabel('Nombre de crimes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'regional_distribution.png'))
    plt.close()
    
    # Statistiques par type de rapport
    print("\nStatistiques par type de rapport :")
    print(region_counts.describe())

def create_crime_map(df, img_dir):
    """Crée une carte des crimes (fonction désactivée car pas de coordonnées)"""
    print("\nCréation de la carte des crimes... (fonction désactivée, pas de coordonnées géographiques)")
    # Cette fonction est désactivée car il n'y a pas de colonnes Latitude/Longitude dans vos données
    pass

def create_heatmap(df, img_dir):
    """Crée une carte de chaleur des crimes (fonction désactivée car pas de coordonnées)"""
    print("\nCréation de la carte de chaleur des crimes... (fonction désactivée, pas de coordonnées géographiques)")
    # Cette fonction est désactivée car il n'y a pas de colonnes Latitude/Longitude dans vos données
    pass

def analyze_category_by_region(df, img_dir):
    """Analyse la distribution des catégories de crimes par type de rapport (Report Type)"""
    print("\nAnalyse de la distribution des catégories de crimes par type de rapport...")
    
    # Distribution des catégories de crimes par type de rapport
    category_region = pd.crosstab(df['Report Type'], df['Category'])
    
    # Visualisation
    plt.figure(figsize=(15, 8))
    category_region.plot(kind='bar', stacked=True)
    plt.title('Distribution des catégories de crimes par type de rapport')
    plt.xlabel('Type de rapport')
    plt.ylabel('Nombre de crimes')
    plt.legend(title='Catégorie', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'category_by_region.png'))
    plt.close()

def main():
    """Fonction principale"""
    print("Démarrage de l'analyse géographique...")
    
    # Création du dossier pour les images
    img_dir = create_img_dir()
    
    # Chargement des données
    df = load_data()
    
    # Exécution des analyses
    analyze_regional_distribution(df, img_dir)
    create_crime_map(df, img_dir)
    create_heatmap(df, img_dir)
    analyze_category_by_region(df, img_dir)
    
    print("\nAnalyse géographique terminée !")
    print(f"Les visualisations ont été sauvegardées dans le dossier : {img_dir}")

if __name__ == "__main__":
    main() 