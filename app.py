from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)

# Chargement des données
df = pd.read_csv('cleaned_crime_data.csv')

# Préparation des données pour la prédiction
def prepare_data():
    features = ['Offender_Age', 'Offender_Gender', 'Offender_Race',
               'Victim_Age', 'Victim_Gender', 'Victim_Race',
               'Report Type']
    
    X = df[features].copy()
    y = df['Category'].copy()
    
    # Encodage des variables catégorielles
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Encodage de la variable cible
    y = le.fit_transform(y)
    
    # Normalisation des variables numériques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, le.classes_, features

# Entraînement du modèle
X_scaled, y, class_names, features = prepare_data()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    return jsonify({
        'categories': df['Category'].value_counts().to_dict(),
        'report_types': df['Report Type'].value_counts().to_dict(),
        'dispositions': df['Disposition'].value_counts().to_dict(),
        'offender_races': df['Offender_Race'].value_counts().to_dict(),
        'victim_races': df['Victim_Race'].value_counts().to_dict(),
        'offender_genders': df['Offender_Gender'].value_counts().to_dict(),
        'victim_genders': df['Victim_Gender'].value_counts().to_dict()
    })

@app.route('/api/filter', methods=['POST'])
def filter_data():
    filters = request.json
    filtered_df = df.copy()
    
    for key, value in filters.items():
        if value and key in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[key] == value]
    
    return jsonify({
        'categories': filtered_df['Category'].value_counts().to_dict(),
        'report_types': filtered_df['Report Type'].value_counts().to_dict(),
        'dispositions': filtered_df['Disposition'].value_counts().to_dict(),
        'offender_races': filtered_df['Offender_Race'].value_counts().to_dict(),
        'victim_races': filtered_df['Victim_Race'].value_counts().to_dict(),
        'offender_genders': filtered_df['Offender_Gender'].value_counts().to_dict(),
        'victim_genders': filtered_df['Victim_Gender'].value_counts().to_dict()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Préparation des données pour la prédiction
    input_data = pd.DataFrame([data])
    
    # Encodage des variables catégorielles
    le = LabelEncoder()
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = le.fit_transform(input_data[col].astype(str))
    
    # Normalisation
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)
    
    # Prédiction
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    return jsonify({
        'prediction': class_names[prediction[0]],
        'probabilities': dict(zip(class_names, probabilities[0]))
    })

if __name__ == '__main__':
    app.run(debug=True) 