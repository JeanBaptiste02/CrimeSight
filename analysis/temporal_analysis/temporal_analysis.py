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

def analyze_category_distribution(df, img_dir):
    """Analyse la distribution des catégories de crimes"""
    print("\nAnalyse de la distribution des catégories de crimes...")
    
    # Distribution des catégories de crimes
    plt.figure(figsize=(12, 6))
    category_counts = df['Category'].value_counts()
    category_counts.plot(kind='bar')
    plt.title('Distribution des catégories de crimes')
    plt.xlabel('Catégorie')
    plt.ylabel('Nombre de crimes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'category_distribution.png'))
    plt.close()
    
    # Statistiques par catégorie
    print("\nStatistiques par catégorie :")
    print(category_counts)

def analyze_report_type_distribution(df, img_dir):
    """Analyse la distribution des types de rapports"""
    print("\nAnalyse de la distribution des types de rapports...")
    
    # Distribution des types de rapports
    plt.figure(figsize=(10, 6))
    report_counts = df['Report Type'].value_counts()
    report_counts.plot(kind='bar')
    plt.title('Distribution des types de rapports')
    plt.xlabel('Type de rapport')
    plt.ylabel('Nombre de rapports')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'report_type_distribution.png'))
    plt.close()
    
    # Statistiques par type de rapport
    print("\nStatistiques par type de rapport :")
    print(report_counts)

def analyze_category_by_report_type(df, img_dir):
    """Analyse la distribution des catégories de crimes par type de rapport"""
    print("\nAnalyse de la distribution des catégories par type de rapport...")
    
    # Distribution croisée
    plt.figure(figsize=(15, 8))
    cross_tab = pd.crosstab(df['Report Type'], df['Category'])
    cross_tab.plot(kind='bar', stacked=True)
    plt.title('Distribution des catégories de crimes par type de rapport')
    plt.xlabel('Type de rapport')
    plt.ylabel('Nombre de crimes')
    plt.legend(title='Catégorie', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'category_by_report_type.png'))
    plt.close()
    
    # Statistiques croisées
    print("\nStatistiques croisées :")
    print(cross_tab)

def analyze_disposition_distribution(df, img_dir):
    """Analyse la distribution des dispositions"""
    print("\nAnalyse de la distribution des dispositions...")
    
    # Distribution des dispositions
    plt.figure(figsize=(10, 6))
    disposition_counts = df['Disposition'].value_counts()
    disposition_counts.plot(kind='bar')
    plt.title('Distribution des dispositions')
    plt.xlabel('Disposition')
    plt.ylabel('Nombre de cas')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'disposition_distribution.png'))
    plt.close()
    
    # Statistiques par disposition
    print("\nStatistiques par disposition :")
    print(disposition_counts)

def main():
    """Fonction principale"""
    print("Démarrage de l'analyse temporelle...")
    
    # Création du dossier pour les images
    img_dir = create_img_dir()
    
    # Chargement des données
    df = load_data()
    
    # Exécution des analyses
    analyze_category_distribution(df, img_dir)
    analyze_report_type_distribution(df, img_dir)
    analyze_category_by_report_type(df, img_dir)
    analyze_disposition_distribution(df, img_dir)
    
    print("\nAnalyse terminée !")
    print(f"Les visualisations ont été sauvegardées dans le dossier : {img_dir}")

if __name__ == "__main__":
    main() 