#!/usr/bin/env python
# coding: utf-8

# # Analyse Exploratoire des Données (EDA) - Projet Telco Customer Churn
# 
# Ce notebook est dédié à l'analyse exploratoire des données (EDA) du dataset Telco Customer Churn. L'objectif est de comprendre la structure des données, d'identifier les caractéristiques importantes et les tendances liées au désabonnement des clients (churn), et d'évaluer la qualité du dataset avant les étapes de prétraitement et de modélisation.
# 
# ---

# ### 1. Initialisation de l'Environnement et Chargement des Données
# 
# Cette section prépare notre environnement de travail en important les bibliothèques Python nécessaires et en chargeant le dataset.

# In[1]:


# Cell 1: Import des bibliothèques nécessaires

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pour ydata-profiling
from ydata_profiling import ProfileReport


# #### 1.1. Chargement du Dataset
# 
# Nous chargeons le dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` dans un DataFrame Pandas. Ce dataset contient des informations sur les clients d'une entreprise de télécommunications et leur statut de désabonnement (Churn), qui est notre variable cible. Cette étape marque le début de la phase de collecte et d'évaluation des données.

# In[3]:


# Cell 2: Chargement du dataset - UTILISER CE CHEMIN RELATIF POUR LA PORTABILITÉ ET GITHUB !
try:
    # Le notebook est dans 'notebooks/', le fichier est dans '../data/'
    df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Dataset chargé avec succès !")
except FileNotFoundError:
    print("Erreur : Le fichier CSV n'a pas été trouvé. Vérifiez le chemin d'accès relatif.")
    print("Assurez-vous que 'WA_Fn-UseC_-Telco-Customer-Churn.csv' est bien dans le dossier 'data/'")
    print("et que votre notebook est lancé depuis le dossier parent 'Projet_Churn_Telco/'.")
    exit()


# ### 2. Aperçu Initial et Nettoyage de Base du Dataset
# 
# Cette section fournit un premier aperçu des données (premières lignes, dimensions, types de colonnes) et effectue un nettoyage initial essentiel.

# #### 2.1. Affichage des Premières Lignes et de la Forme du Dataset
# 
# Nous commençons par inspecter les premières lignes du DataFrame pour comprendre la structure générale des données et vérifier les dimensions du dataset (nombre de lignes et de colonnes).

# In[5]:


# Cell 3: Afficher les premières lignes et la forme du dataset
print("--- Aperçu des 5 premières lignes du dataset ---")
print(df.head())
print("\n--- Dimensions du dataset (lignes, colonnes) ---")
print(df.shape)


# #### 2.2. Informations Générales sur les Colonnes et Types de Données
# 
# La fonction `df.info()` nous donne un résumé concis du DataFrame, incluant le nombre d'entrées non nulles pour chaque colonne et leur type de données. Cela est crucial pour identifier les colonnes qui nécessitent un nettoyage ou une conversion de type.

# In[7]:


# Cell 4: Informations générales sur le dataset
print("\n--- Informations sur les colonnes et types de données ---")
df.info()


# #### Interprétation des Cellules 3 & 4 : Vue d'ensemble du Dataset
# 
# Les premières lignes du dataset (`df.head()`) nous donnent un aperçu des colonnes et de quelques valeurs. Le dataset contient `7043` lignes (clients) et `21` colonnes. La fonction `df.info()` révèle que la plupart des colonnes sont de type `object` (catégorielles), avec `tenure` et `MonthlyCharges` comme `int64`/`float64`. Une observation cruciale est que `TotalCharges` est de type `object`, indiquant la présence de valeurs non numériques nécessitant un nettoyage.

# #### 2.3. Nettoyage de la colonne 'TotalCharges'
# 
# Une observation initiale importante est que la colonne `TotalCharges` est de type `object`, ce qui indique qu'elle contient des valeurs non numériques (probablement des espaces ou des chaînes vides) qui devront être converties en numérique. Nous identifions ces valeurs non numériques et les imputons avec la médiane de la colonne. Cette étape est cruciale pour permettre des calculs numériques et des visualisations ultérieures sur cette variable.

# In[9]:


# Cell 5: Vérification des valeurs manquantes
print("\n--- Nombre de valeurs manquantes par colonne ---")
print(df.isnull().sum())

# Il est fort probable que 'TotalCharges' contienne des espaces vides qui sont interprétés comme des objets (chaînes de caractères)
# et non comme des valeurs numériques par pandas. Il faut les convertir.
print("\n--- Traitement de la colonne 'TotalCharges' ---")
# Remplacer les espaces vides par NaN, puis convertir en numérique
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Vérifier à nouveau les valeurs manquantes après conversion
print("Valeurs manquantes dans 'TotalCharges' après conversion :", df['TotalCharges'].isnull().sum())

# Gérer les valeurs manquantes dans 'TotalCharges' (par exemple, remplacer par la médiane ou la moyenne)
# Pour l'EDA, nous allons simplement les remplir pour pouvoir les analyser, mais lors du prétraitement, nous affinerons.
# Ici, nous allons les remplir avec la médiane pour ne pas fausser les analyses statistiques simples.
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print("Valeurs manquantes dans 'TotalCharges' après remplissage :", df['TotalCharges'].isnull().sum())


# #### Interprétation de la Cellule 5 : Nettoyage de 'TotalCharges'
# 
# La colonne `TotalCharges` a été correctement convertie en type `float64` après avoir géré les `11` valeurs non numériques (qui étaient probablement des chaînes vides) en les remplaçant par la médiane de la colonne. Cette étape est fondamentale pour permettre des analyses numériques et une utilisation correcte de cette variable dans les phases ultérieures.

# ### 3. Rapport de Profilage Détaillé du Dataset
# 
# Pour une analyse exploratoire approfondie et automatisée, nous utilisons la bibliothèque `ydata-profiling`. Ce rapport HTML interactif fournit un inventaire complet du dataset, incluant des statistiques descriptives pour chaque variable, des distributions, des corrélations, des informations sur les valeurs manquantes et les doublons. Cet inventaire est fondamental pour comprendre la structure et la qualité des données.

# In[11]:


# Cell 6: Générer le rapport de profilage avec ydata-profiling
print("\n--- Génération du rapport de profilage (cela peut prendre quelques instants) ---")
profile = ProfileReport(df, title="Telco Customer Churn - Rapport d'Analyse Exploratoire")
profile.to_file("../documentation/Profiling_Report_Telco_Churn.html")
print("Rapport de profilage généré et sauvegardé dans 'documentation/Profiling_Report_Telco_Churn.html'")


# ### 4. Analyse de la Variable Cible 'Churn'
# 
# Cette section se concentre sur la distribution de la variable `Churn` pour comprendre la proportion de clients qui se sont désabonnés. Cela est crucial pour évaluer le déséquilibre des classes, un aspect important pour la modélisation ultérieure.

# In[13]:


# Cell 7: Analyse de la variable cible 'Churn'
print("\n--- Distribution de la variable cible 'Churn' ---")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True) * 100)

sns.countplot(x='Churn', data=df)
plt.title('Distribution du Churn')
plt.show()


# #### Interprétation de la Cellule 7 : Distribution du Churn
# 
# La distribution de la variable cible `Churn` révèle un déséquilibre significatif :
# - `5174` clients (`73.46%`) n'ont pas churné (`No`).
# - `1869` clients (`26.54%`) ont churné (`Yes`).
# Ce déséquilibre (environ 73% 'No' vs 27% 'Yes') est important à noter et devra être considéré lors des étapes de modélisation pour éviter que le modèle ne soit biaisé en faveur de la classe majoritaire.

# ### 5. Analyse de l'Impact des Variables Numériques sur le Churn
# 
# Nous explorons ici la relation entre les variables numériques clés (`tenure`, `MonthlyCharges`, `TotalCharges`) et le statut de `Churn`. Les histogrammes nous aident à visualiser la distribution de ces variables pour chaque catégorie de churn, tandis que les box plots mettent en évidence les médianes et la dispersion.

# In[15]:


# Cell 8: Analyse des variables numériques clés par rapport au Churn
print("\n--- Analyse des variables numériques par rapport au Churn ---")
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(data=df, x=col, hue='Churn', kde=True, palette='coolwarm')
    plt.title(f'Distribution de {col} par Churn')
plt.tight_layout()
plt.show()

# Box plots pour les variables numériques (comme suggéré dans votre capture)
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x='Churn', y=col, data=df, palette='viridis')
    plt.title(f'Box plot de {col} par Churn')
plt.tight_layout()
plt.show()


# #### Interprétation de la Cellule 8 : Impact des Numériques sur le Churn
# 
# Les visualisations des variables numériques par rapport au `Churn` révèlent des tendances clés :
# - **`tenure` (Ancienneté) :** Les clients qui churnent (`Yes`) ont tendance à avoir une ancienneté beaucoup plus faible, avec une forte concentration de churners parmi les nouveaux clients. Les clients fidèles (`No`) sont généralement ceux ayant une longue ancienneté.
# - **`MonthlyCharges` (Frais Mensuels) :** Les clients avec des frais mensuels plus élevés montrent une probabilité plus grande de churn.
# - **`TotalCharges` (Frais Totaux) :** Les clients qui churnent ont des frais totaux plus faibles. Cela est directement lié à leur faible ancienneté (`tenure`), car les frais totaux s'accumulent avec le temps.

# ### 6. Analyse de l'Impact des Variables Catégorielles sur le Churn
# 
# Cette section explore la relation entre chaque variable catégorielle et le statut de `Churn` à l'aide de graphiques de comptage (`countplot`). Cela permet d'identifier visuellement les catégories qui sont plus ou moins susceptibles de mener au désabonnement.

# In[17]:


# Cell 9: Analyse des variables catégorielles clés par rapport au Churn
print("\n--- Analyse des variables catégorielles par rapport au Churn ---")
# Exclure customerID et les colonnes numériques déjà traitées, ainsi que Churn
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('customerID') # On retire l'ID client
if 'Churn' in categorical_cols:
    categorical_cols.remove('Churn') # On retire Churn si c'est encore un objet

plt.figure(figsize=(20, 25))
for i, col in enumerate(categorical_cols):
    plt.subplot(5, 4, i + 1) # Ajustez les dimensions si vous avez plus de colonnes
    sns.countplot(data=df, x=col, hue='Churn', palette='pastel')
    plt.title(f'Distribution de {col} par Churn')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# #### Interprétation de la Cellule 9 : Impact des Catégorielles sur le Churn
# 
# L'analyse des variables catégorielles par rapport au `Churn` fournit des insights essentiels :
# - **`Contract` (Type de Contrat) :** Les clients avec un contrat `Month-to-month` (mensuel) ont un taux de churn massivement plus élevé que ceux avec des contrats d'un an ou de deux ans. C'est un facteur très influent.
# - **`InternetService` (Service Internet) :** Les abonnés à la `Fiber optic` montrent un taux de churn nettement supérieur aux abonnés `DSL`.
# - **Services Additionnels (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport) :** Les clients qui n'ont PAS ces services sont plus susceptibles de churner. Ces services semblent augmenter la fidélité.
# - **`Partner` (Partenaire) et `Dependents` (Personnes à charge) :** Les clients sans partenaire et sans personnes à charge sont plus à risque de désabonnement.
# - **`PaperlessBilling` (Facturation sans papier) :** Les clients avec la facturation sans papier ont un taux de churn plus élevé.
# - **`PaymentMethod` (Méthode de Paiement) :** Le paiement par `Electronic check` est fortement associé au churn.
# - **`gender` (Genre) et `PhoneService` (Service Téléphonique) :** Ces variables semblent avoir un impact minimal sur le churn.

# ### 7. Corrélation entre les Variables Numériques
# 
# Cette section explore les relations linéaires entre les variables numériques du dataset, y compris la variable cible `Churn` après l'avoir convertie en format numérique (0 pour 'No', 1 pour 'Yes'). Une matrice de corrélation et un heatmap sont utilisés pour visualiser ces relations.

# In[19]:


# Cell 10: Corrélation entre les variables numériques (si besoin pour insights)
print("\n--- Matrice de corrélation des variables numériques ---")
# Pour la corrélation de Churn, il faut le convertir en numérique (oui=1, non=0)
df['Churn_numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print("\n--- Matrice de corrélation incluant Churn (numérique) ---")
print(df[numeric_cols + ['Churn_numeric']].corr())

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols + ['Churn_numeric']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de Corrélation')
plt.show()


# #### Interprétation de la Cellule 10 : Corrélations Numériques
# 
# La matrice de corrélation confirme plusieurs relations :
# - Une forte corrélation positive (`0.83`) est observée entre `tenure` et `TotalCharges`, ce qui est logique car les frais totaux augmentent avec l'ancienneté du client.
# - `MonthlyCharges` et `TotalCharges` sont modérément corrélées positivement (`0.65`).
# - Une corrélation négative modérée (`-0.35`) entre `tenure` et `Churn_numeric` indique que les clients avec une faible ancienneté sont plus susceptibles de churner.
# - Les corrélations entre `MonthlyCharges` et `Churn_numeric` (`0.19`), et `TotalCharges` et `Churn_numeric` (`-0.20`), sont plus faibles mais cohérentes avec les observations faites lors des analyses individuelles des variables.

# In[ ]:




