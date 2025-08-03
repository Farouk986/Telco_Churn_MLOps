#!/usr/bin/env python
# coding: utf-8

# Analyse Exploratoire des Données (EDA) - Projet Telco Customer Churn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du dataset (chemin relatif)
try:
    df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Dataset chargé avec succès !")
except FileNotFoundError:
    print("Erreur : Le fichier CSV n'a pas été trouvé. Vérifiez le chemin d'accès relatif.")
    exit()

# Aperçu initial
print("--- Aperçu des 5 premières lignes du dataset ---")
print(df.head())
print("\n--- Dimensions du dataset (lignes, colonnes) ---")
print(df.shape)

print("\n--- Informations sur les colonnes et types de données ---")
df.info()

# Nettoyage de la colonne 'TotalCharges'
print("\n--- Nombre de valeurs manquantes par colonne ---")
print(df.isnull().sum())

print("\n--- Traitement de la colonne 'TotalCharges' ---")
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
print("Valeurs manquantes dans 'TotalCharges' après conversion :", df['TotalCharges'].isnull().sum())

df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print("Valeurs manquantes dans 'TotalCharges' après remplissage :", df['TotalCharges'].isnull().sum())

# Analyse de la variable cible 'Churn'
print("\n--- Distribution de la variable cible 'Churn' ---")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True) * 100)

sns.countplot(x='Churn', data=df)
plt.title('Distribution du Churn')
plt.show()

# Analyse des variables numériques clés par rapport au Churn
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(data=df, x=col, hue='Churn', kde=True, palette='coolwarm')
    plt.title(f'Distribution de {col} par Churn')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x='Churn', y=col, data=df, palette='viridis')
    plt.title(f'Box plot de {col} par Churn')
plt.tight_layout()
plt.show()

# Analyse des variables catégorielles clés par rapport au Churn
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('customerID')
if 'Churn' in categorical_cols:
    categorical_cols.remove('Churn')

plt.figure(figsize=(20, 25))
for i, col in enumerate(categorical_cols):
    plt.subplot(5, 4, i + 1)
    sns.countplot(data=df, x=col, hue='Churn', palette='pastel')
    plt.title(f'Distribution de {col} par Churn')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Corrélation entre les variables numériques
df['Churn_numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print("\n--- Matrice de corrélation incluant Churn (numérique) ---")
print(df[numeric_cols + ['Churn_numeric']].corr())

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols + ['Churn_numeric']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de Corrélation')
plt.show()
