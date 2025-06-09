import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Configuration de l'affichage
plt.style.use('default')  # Utilisation du style par défaut
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

def prepare_data(df):
    """Prépare les données pour la modélisation"""
    print("\nPréparation des données pour la modélisation...")
    
    # Sélection des caractéristiques pour la prédiction
    features = ['Offender_Age', 'Offender_Gender', 'Offender_Race',
               'Victim_Age', 'Victim_Gender', 'Victim_Race',
               'Report Type', 'Category']
    
    # Création d'une copie des données pour éviter les SettingWithCopyWarning
    X = df[features].copy()
    y = df['Category'].copy()
    
    # Affichage de la distribution initiale des classes
    print("\nDistribution initiale des classes :")
    print(y.value_counts())
    
    # Suppression des classes avec moins de 10 exemples
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    
    print("\nDistribution des classes après filtrage :")
    print(y.value_counts())
    
    # Encodage des variables catégorielles
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Encodage de la variable cible
    y = le.fit_transform(y)
    
    # Normalisation des variables numériques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"\nTaille de l'ensemble d'entraînement : {X_train.shape[0]}")
    print(f"Taille de l'ensemble de test : {X_test.shape[0]}")
    
    # Vérification des classes présentes
    print("\nClasses présentes dans les données :")
    for i, class_name in enumerate(le.classes_):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"{class_name}: {train_count} (train) / {test_count} (test)")
    
    return X_train, X_test, y_train, y_test, le.classes_

def train_random_forest(X_train, X_test, y_train, y_test, img_dir, class_names):
    """Entraîne et évalue un modèle Random Forest"""
    print("\nEntraînement du modèle Random Forest...")
    
    # Entraînement du modèle
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_rf = rf_model.predict(X_test)
    
    # Évaluation du modèle
    print("\nRapport de classification pour Random Forest :")
    print(classification_report(y_test, y_pred_rf, 
                              target_names=class_names,
                              labels=np.unique(y_test)))  # Utilise uniquement les classes présentes
    
    # Matrice de confusion
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[np.unique(y_test)],
                yticklabels=class_names[np.unique(y_test)])
    plt.title('Matrice de confusion - Random Forest')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs réelles')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'rf_confusion_matrix.png'))
    plt.close()
    
    return rf_model, y_pred_rf

def train_xgboost(X_train, X_test, y_train, y_test, img_dir, class_names):
    """Entraîne et évalue un modèle XGBoost"""
    print("\nEntraînement du modèle XGBoost...")
    
    # Entraînement du modèle
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Évaluation du modèle
    print("\nRapport de classification pour XGBoost :")
    print(classification_report(y_test, y_pred_xgb, 
                              target_names=class_names,
                              labels=np.unique(y_test)))  # Utilise uniquement les classes présentes
    
    # Matrice de confusion
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_xgb)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[np.unique(y_test)],
                yticklabels=class_names[np.unique(y_test)])
    plt.title('Matrice de confusion - XGBoost')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs réelles')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'xgb_confusion_matrix.png'))
    plt.close()
    
    return xgb_model, y_pred_xgb

def compare_models(y_test, y_pred_rf, y_pred_xgb, img_dir):
    """Compare les performances des modèles"""
    print("\nComparaison des performances des modèles...")
    
    # Calcul des métriques pour les deux modèles
    metrics_data = []
    
    for name, preds in [('Random Forest', y_pred_rf), ('XGBoost', y_pred_xgb)]:
        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, average='weighted'),
            'Recall': recall_score(y_test, preds, average='weighted'),
            'F1': f1_score(y_test, preds, average='weighted')
        })
    
    metrics = pd.DataFrame(metrics_data)
    
    print("\nComparaison des performances des modèles :")
    print(metrics)
    
    # Visualisation des métriques
    plt.figure(figsize=(12, 6))
    metrics.set_index('Model').plot(kind='bar')
    plt.title('Comparaison des performances des modèles')
    plt.xlabel('Modèle')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'model_comparison.png'))
    plt.close()

def analyze_feature_importance(rf_model, features, img_dir):
    """Analyse l'importance des caractéristiques"""
    print("\nAnalyse de l'importance des caractéristiques...")
    
    # Importance des caractéristiques pour Random Forest
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Importance des caractéristiques - Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'feature_importance.png'))
    plt.close()

def main():
    """Fonction principale"""
    print("Démarrage de l'analyse prédictive...")
    
    # Création du dossier pour les images
    img_dir = create_img_dir()
    
    # Chargement des données
    df = load_data()
    
    # Préparation des données
    X_train, X_test, y_train, y_test, class_names = prepare_data(df)
    
    # Entraînement et évaluation des modèles
    rf_model, y_pred_rf = train_random_forest(X_train, X_test, y_train, y_test, img_dir, class_names)
    xgb_model, y_pred_xgb = train_xgboost(X_train, X_test, y_train, y_test, img_dir, class_names)
    
    # Comparaison des modèles
    compare_models(y_test, y_pred_rf, y_pred_xgb, img_dir)
    
    # Analyse de l'importance des caractéristiques
    features = ['Offender_Age', 'Offender_Gender', 'Offender_Race',
               'Victim_Age', 'Victim_Gender', 'Victim_Race',
               'Report Type', 'Category']
    analyze_feature_importance(rf_model, features, img_dir)
    
    print("\nAnalyse prédictive terminée !")
    print(f"Les visualisations ont été sauvegardées dans le dossier : {img_dir}")

if __name__ == "__main__":
    main() 