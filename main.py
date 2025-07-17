# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:23:31 2025

@author: HAFIAN
"""

# -*- coding: utf-8 -*-
"""
Script Python pour la Prédiction de la Solubilité des Composés Organiques.

Ce script contient les étapes pour :
1. Charger les données des composés organiques.
2. Effectuer une exploration de données simple.
3. Prétraiter les données (sélection des caractéristiques, division, standardisation).
4. Construire et entraîner un modèle de régression linéaire.
5. Évaluer la performance du modèle.
6. Démontrer l'utilisation du modèle pour de nouvelles prédictions.
"""

# --- Importation des Bibliothèques ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

print("Bibliothèques importées avec succès.")

# --- 1. Chargement des Données ---
print("\n--- 1. Chargement des Données ---")
file_path = 'data.csv'

# Vérifier si le fichier data.csv existe, sinon créer un exemple
if not os.path.exists(file_path):
    print(f"Attention: Le fichier '{file_path}' est introuvable. Création d'un fichier d'exemple.")
    sample_data = """SMILES,Solubility,MW,LogP,HBD,HBA
CCO,0.5,46.07,0.01,1,1
CC(=O)Oc1ccccc1C(=O)O,1.2,180.16,1.83,1,3
c1ccccc1,0.001,78.11,2.13,0,0
CCCCC(=O)OC,0.02,116.16,1.4,0,2
O=C(O)c1ccccc1,1.5,122.12,1.3,2,2
CN1CCC(CC1)C(C#N)(c2ccccc2)c3ccccc3,0.0001,312.42,4.8,0,1
NC(CCSCc1ccc(N)cc1)C(=O)O,-2.0,226.3,0.5,3,3
"""
    with open(file_path, 'w') as f:
        f.write(sample_data)
    print("Fichier data.csv créé avec des données d'exemple.")

try:
    df = pd.read_csv(file_path)
    print("Données chargées avec succès. Aperçu des 5 premières lignes :")
    print(df.head())
    print("\nInformations sur les données :")
    df.info()
except Exception as e:
    print(f"Erreur lors du chargement ou de la lecture du fichier CSV : {e}")
    # Quitter si le chargement des données est critique
    exit()

# --- 2. Exploration des Données (EDA) ---
print("\n--- 2. Exploration des Données (EDA) ---")
print("\nStatistiques descriptives des descripteurs numériques :")
print(df.describe())

# Visualisation de la distribution de la solubilité
plt.figure(figsize=(8, 6))
sns.histplot(df['Solubility'], kde=True)
plt.title('Distribution de la Solubilité')
plt.xlabel('Solubilité')
plt.ylabel('Fréquence')
plt.show()

# Visualisation des corrélations entre les descripteurs et la solubilité
numeric_cols = df.select_dtypes(include=np.number).columns
if 'Solubility' not in numeric_cols:
    print("La colonne 'Solubility' n'est pas numérique ou est manquante. Tentative de conversion.")
    df['Solubility'] = pd.to_numeric(df['Solubility'], errors='coerce')
    numeric_cols = df.select_dtypes(include=np.number).columns

if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de Corrélation')
    plt.show()
else:
    print("Pas assez de colonnes numériques pour générer la matrice de corrélation.")


# --- 3. Préparation des Données ---
print("\n--- 3. Préparation des Données ---")

# Sélection des caractéristiques (X) et de la cible (y)
features = [col for col in df.columns if col not in ['SMILES', 'Solubility']]
if not features:
    print("Erreur: Aucune colonne de descripteurs numériques trouvée pour l'entraînement du modèle.")
    print("Veuillez vérifier votre fichier data.csv et le nom des colonnes.")
    exit()

X = df[features]
y = df['Solubility']

# Vérification et gestion des valeurs manquantes
print(f"\nValeurs manquantes dans X avant prétraitement:\n{X.isnull().sum()}")
print(f"\nValeurs manquantes dans y avant prétraitement:\n{y.isnull().sum()}")

# Imputation simple (remplir les NaN avec la moyenne)
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].mean())
if y.isnull().any():
    y = y.fillna(y.mean())
print("\nValeurs manquantes traitées (imputation par la moyenne).")

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTaille de l'ensemble d'entraînement (X_train): {X_train.shape}")
print(f"Taille de l'ensemble de test (X_test): {X_test.shape}")

# Standardisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reconvertir en DataFrame pour conserver les noms de colonnes pour la lisibilité
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
print("Caractéristiques standardisées.")

# --- 4. Construction et Entraînement du Modèle ---
print("\n--- 4. Construction et Entraînement du Modèle ---")
model = LinearRegression()
model.fit(X_train_scaled_df, y_train)

print("\nModèle de Régression Linéaire entraîné.")
print(f"Coefficients du modèle: {model.coef_}")
print(f"Ordonnée à l'origine (intercept): {model.intercept_}")

# --- 5. Évaluation du Modèle ---
print("\n--- 5. Évaluation du Modèle ---")
y_pred = model.predict(X_test_scaled_df)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Performance du Modèle ---")
print(f"Erreur Quadratique Moyenne (MSE): {mse:.4f}")
print(f"Racine de l'Erreur Quadratique Moyenne (RMSE): {rmse:.4f}")
print(f"Coefficient de Détermination (R²): {r2:.4f}")

# Visualisation des prédictions vs les vraies valeurs
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Prédiction parfaite')
plt.xlabel('Solubilité Réelle')
plt.ylabel('Solubilité Prédite')
plt.title('Prédictions du Modèle vs Solubilité Réelle')
plt.legend()
plt.grid(True)
plt.show()

# Visualisation des résidus
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution des Résidus')
plt.xlabel('Résidus (Solubilité Réelle - Prédite)')
plt.ylabel('Fréquence')
plt.show()

# --- 6. Utilisation du Modèle (Exemple de Prédiction) ---
print("\n--- 6. Utilisation du Modèle (Exemple de Prédiction) ---")

# Exemple de descripteurs pour un nouveau composé à prédire
# Assurez-vous que l'ordre des descripteurs est le même que celui utilisé pour l'entraînement (features)
# Pour cet exemple: MW, LogP, HBD, HBA
new_compound_data = np.array([[120, 0.8, 1, 3]])
new_compound_df = pd.DataFrame(new_compound_data, columns=features)

# Appliquer le même scaler utilisé pour l'entraînement
new_compound_scaled = scaler.transform(new_compound_df)

# Faire la prédiction
predicted_solubility = model.predict(new_compound_scaled)[0]

print(f"Solubilité prédite pour le nouveau composé avec les descripteurs {new_compound_df.values[0]}: {predicted_solubility:.4f}")

# --- 7. Améliorations Possibles ---
print("\n--- 7. Améliorations Possibles ---")
print("- **Plus de données :** Utiliser un jeu de données plus grand et plus diversifié.")
print("- **Plus de descripteurs :** Générer un ensemble plus riche de descripteurs moléculaires (e.g., via RDKit, Mordred).")
print("- **Autres modèles :** Tester différents algorithmes de régression (Random Forest, Gradient Boosting, SVM, réseaux de neurones).")
print("- **Validation croisée :** Utiliser la validation croisée pour une évaluation plus robuste du modèle.")
print("- **Optimisation des hyperparamètres :** Utiliser GridSearchCV ou RandomizedSearchCV.")
print("- **Déploiement :** Sauvegarder le modèle entraîné pour une utilisation ultérieure (e.g., avec `joblib` ou `pickle`).")