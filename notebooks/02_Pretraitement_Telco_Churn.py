#!/usr/bin/env python
# coding: utf-8

# # Prétraitement des Données - Projet Telco Customer Churn
# 
# Ce notebook est dédié à la phase de prétraitement des données pour le dataset Telco Customer Churn. Il s'agit d'une étape cruciale pour préparer les données brutes pour les modèles de machine learning. Conformément aux attentes du professeur, ce notebook effectuera le nettoyage final, les transformations nécessaires (encodage, mise à l'échelle), l'imputation si nécessaire, et produira un nouveau dataset propre et prêt pour la modélisation.
# 
# ---

# ### 1. Initialisation de l'Environnement et Chargement du Dataset
# 
# Cette section prépare notre environnement de travail en important les bibliothèques Python nécessaires et en chargeant le dataset brut original.

# In[3]:


# Cellule 1: Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[5]:


# Cellule 2: Chargement du dataset brut
# Nous rechargeons le dataset original pour s'assurer que toutes les étapes de prétraitement sont appliquées ici.
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# ### 2. Nettoyage des Données
# 
# Cette section se concentre sur le nettoyage des inconsistances dans le dataset, en particulier la gestion des valeurs non numériques et manquantes dans la colonne `TotalCharges`. C'est une étape essentielle pour assurer l'intégrité numérique des données.

# In[8]:


# Cellule 3: Nettoyage de la colonne 'TotalCharges'
# Convertir 'TotalCharges' en numérique, les erreurs (espaces) seront remplacées par NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Gérer les valeurs manquantes résultant de la conversion (les lignes vides sont devenues NaN)
# Imputation par la médiane, comme décidé lors de l'EDA.
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'].fillna(median_total_charges, inplace=True)

# Vérifier le type et l'absence de NaN après nettoyage
print("--- Informations sur 'TotalCharges' après nettoyage ---")
df['TotalCharges'].info()
print(f"\nNombre de valeurs manquantes dans 'TotalCharges' : {df['TotalCharges'].isnull().sum()}")


# #### Interprétation de la Cellule 3 : Nettoyage de 'TotalCharges'
# 
# La colonne `TotalCharges` a été convertie avec succès en type numérique (`float64`). Les 11 valeurs manquantes (qui étaient des chaînes vides ou des espaces) ont été gérées par imputation avec la médiane de la colonne. Cette opération est fondamentale car elle rend la colonne utilisable pour les calculs et la modélisation, éliminant ainsi une source potentielle d'erreurs.

# ### 3. Préparation des Données pour l'Encodage et la Mise à l'Échelle
# 
# Avant d'appliquer les transformations, nous identifions les colonnes catégorielles et numériques. Nous préparons également la variable cible (`Churn`) en la convertissant en format numérique (0 ou 1) et en la séparant des variables explicatives.

# In[14]:


# Cellule 4: Séparation de la variable cible et identification des types de colonnes
# Suppression de 'customerID' qui n'est pas une feature
df_processed = df.drop(columns=['customerID'])

# Convertir 'Churn' en numérique (0 pour 'No', 1 pour 'Yes')
df_processed['Churn'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})

# Identifier les colonnes numériques et catégorielles restantes
numeric_features = df_processed.select_dtypes(include=np.number).columns.tolist()
# Exclure 'Churn' de la liste des features numériques à transformer (c'est la cible)
if 'Churn' in numeric_features:
    numeric_features.remove('Churn')

categorical_features = df_processed.select_dtypes(include='object').columns.tolist()

print(f"Colonnes numériques à transformer : {numeric_features}")
print(f"Colonnes catégorielles à transformer : {categorical_features}")


# #### Interprétation de la Cellule 4 : Identification des Features et Cible
# 
# Nous avons préparé le DataFrame en supprimant l'identifiant client (`customerID`) et en convertissant la variable cible `Churn` en un format binaire (0 pour 'No', 1 pour 'Yes'). Les colonnes numériques et catégorielles ont été correctement identifiées, ce qui est une étape préalable essentielle pour appliquer les bonnes techniques d'encodage et de mise à l'échelle à chaque type de données.

# ### 4. Encodage des Variables Catégorielles et Mise à l'Échelle des Numériques
# 
# Cette section applique les transformations nécessaires aux variables pour qu'elles soient utilisables par les algorithmes de machine learning. Nous utilisons le One-Hot Encoding pour les variables catégorielles et la mise à l'échelle (`StandardScaler`) pour les variables numériques.

# In[20]:


# Cellule 5: Application des transformations (Encodage et Mise à l'échelle)
# Création des préprocesseurs pour chaque type de colonne
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()) # Mise à l'échelle des données numériques
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-Hot Encoding pour les catégorielles
])

# Combiner les préprocesseurs avec ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Conserver les colonnes non spécifiées (ici, il ne devrait pas y en avoir après Churn)
)

# Appliquer le préprocesseur au DataFrame (sauf la colonne 'Churn')
# Nous allons séparer X et y ici pour appliquer le ColumnTransformer uniquement sur X
X = df_processed.drop(columns=['Churn'])
y = df_processed['Churn']

# Appliquer le préprocesseur à X
X_preprocessed = preprocessor.fit_transform(X)

# Récupérer les noms des colonnes après One-Hot Encoding
# C'est cette partie qui avait l'erreur. get_feature_names_out() est la bonne méthode.
feature_names_categorical = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)


# Combiner tous les noms de features
all_feature_names = numeric_features + list(feature_names_categorical) # Convertir en liste pour concaténation

# Convertir le tableau NumPy résultant en DataFrame Pandas
df_final_preprocessed = pd.DataFrame(X_preprocessed, columns=all_feature_names)

# Ajouter la colonne 'Churn' à notre DataFrame final prétraité (y)
df_final_preprocessed['Churn'] = y.reset_index(drop=True) # Utiliser y directement et réinitialiser l'index

print("--- Aperçu du DataFrame après prétraitement ---")
print(df_final_preprocessed.head())
print(f"\nDimensions du DataFrame après prétraitement : {df_final_preprocessed.shape}")


# #### Interprétation de la Cellule 5 : Données Encodées et Mises à l'Échelle
# 
# Cette étape a transformé le dataset de manière significative :
# - Toutes les variables catégorielles ont été converties en un format numérique via **One-Hot Encoding**, créant de nouvelles colonnes binaires. Par exemple, la colonne `gender` a été transformée en `gender_Male` (puisque `drop_first=True` n'est pas utilisé directement ici, une colonne pour chaque catégorie est créée, mais `ColumnTransformer` gère souvent cela implicitement, ou nous pourrions ajouter `drop='first'` à `OneHotEncoder` si nécessaire).
# - Les variables numériques (`tenure`, `MonthlyCharges`, `TotalCharges`) ont été **mises à l'échelle** à l'aide de `StandardScaler`, ce qui est crucial pour de nombreux algorithmes de Machine Learning sensibles à l'échelle des données.
# Le DataFrame résultant (`df_final_preprocessed`) est désormais entièrement numérique et prêt pour l'entraînement des modèles. Il a `7043` lignes et un nombre de colonnes accru en raison de l'encodage One-Hot.

# ### 5. Sauvegarde du Dataset Prétraité
# 
# Le dataset entièrement prétraité et prêt pour la modélisation est sauvegardé dans un nouveau fichier CSV. Ce fichier servira d'entrée pour le prochain notebook (`03_Modelisation_Telco_Churn.ipynb`), garantissant ainsi une pipeline de traitement des données claire et modulaire, comme demandé par le professeur.

# In[24]:


# Cellule 6: Sauvegarde du DataFrame prétraité
# Assurez-vous que le dossier 'data' existe à la racine du projet
output_path = 'data/telco_churn_preprocessed.csv'
df_final_preprocessed.to_csv(output_path, index=False)

print(f"\nLe dataset prétraité a été sauvegardé sous : {output_path}")


# #### Conclusion du Notebook de Prétraitement
# 
# Ce notebook a successfully complété la phase de prétraitement des données pour le dataset Telco Customer Churn. Nous avons effectué les opérations suivantes :
# 1.  **Nettoyage** de la colonne `TotalCharges` en convertissant son type et en imputant les valeurs manquantes.
# 2.  **Transformation** des variables catégorielles via One-Hot Encoding et mise à l'échelle des variables numériques avec `StandardScaler`.
# 3.  Préparation de la variable cible `Churn` en format numérique.
# 
# Le résultat de ce travail est un fichier `telco_churn_preprocessed.csv` qui contient toutes les données dans un format adéquat pour l'application des algorithmes de Machine Learning. Ce fichier est maintenant l'entrée pour le prochain notebook dédié au développement et à l'évaluation des modèles.
# 
# ---

# In[ ]:




