import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration de l'affichage
plt.style.use('default')  # Utilisation du style par défaut au lieu de seaborn
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
    df = pd.read_csv('../../cleaned_crime_data.csv')  # Chemin corrigé
    print(f"Nombre total d'enregistrements : {len(df)}")
    return df

def analyze_offender_victim_relationship(df, img_dir):
    """Analyse la relation entre les caractéristiques des auteurs et des victimes"""
    print("\nAnalyse de la relation entre les auteurs et les victimes...")
    
    # Relation entre l'âge de l'auteur et l'âge de la victime
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Offender_Age'], df['Victim_Age'], alpha=0.5)
    plt.title('Relation entre l\'âge de l\'auteur et l\'âge de la victime')
    plt.xlabel('Âge de l\'auteur')
    plt.ylabel('Âge de la victime')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'age_relationship.png'))
    plt.close()
    
    # Distribution des crimes par genre de l'auteur et de la victime
    plt.figure(figsize=(12, 6))
    gender_cross = pd.crosstab(df['Offender_Gender'], df['Victim_Gender'])
    gender_cross.plot(kind='bar', stacked=True)
    plt.title('Distribution des crimes par genre de l\'auteur et de la victime')
    plt.xlabel('Genre de l\'auteur')
    plt.ylabel('Nombre de crimes')
    plt.legend(title='Genre de la victime')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'gender_relationship.png'))
    plt.close()
    
    # Distribution des crimes par race de l'auteur et de la victime
    plt.figure(figsize=(15, 8))
    race_cross = pd.crosstab(df['Offender_Race'], df['Victim_Race'])
    race_cross.plot(kind='bar', stacked=True)
    plt.title('Distribution des crimes par race de l\'auteur et de la victime')
    plt.xlabel('Race de l\'auteur')
    plt.ylabel('Nombre de crimes')
    plt.legend(title='Race de la victime', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'race_relationship.png'))
    plt.close()

def analyze_crime_patterns(df, img_dir):
    """Analyse les patterns de crimes"""
    print("\nAnalyse des patterns de crimes...")
    
    # Distribution des crimes par catégorie
    plt.figure(figsize=(12, 6))
    df['Category'].value_counts().plot(kind='bar')
    plt.title('Distribution des crimes par catégorie')
    plt.xlabel('Catégorie de crime')
    plt.ylabel('Nombre de crimes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'crime_categories.png'))
    plt.close()
    
    # Relation entre la catégorie de crime et le statut de l'auteur
    plt.figure(figsize=(15, 8))
    status_cross = pd.crosstab(df['Category'], df['OffenderStatus'])
    status_cross.plot(kind='bar', stacked=True)
    plt.title('Distribution des crimes par catégorie et statut de l\'auteur')
    plt.xlabel('Catégorie de crime')
    plt.ylabel('Nombre de crimes')
    plt.legend(title='Statut de l\'auteur', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'crime_status.png'))
    plt.close()

def main():
    """Fonction principale"""
    print("Démarrage de l'analyse des relations...")
    
    # Création du dossier pour les images
    img_dir = create_img_dir()
    
    # Chargement des données
    df = load_data()
    
    # Exécution des analyses
    analyze_offender_victim_relationship(df, img_dir)
    analyze_crime_patterns(df, img_dir)
    
    print("\nAnalyse des relations terminée !")
    print(f"Les visualisations ont été sauvegardées dans le dossier : {img_dir}")

if __name__ == "__main__":
    main() 