#!/usr/bin/env python
# coding: utf-8

# # Modélisation et Évaluation - Projet Telco Customer Churn
# 
# Ce notebook est dédié à la phase de modélisation et d'évaluation pour le dataset Telco Customer Churn. Il constitue la dernière étape majeure du pipeline de machine learning. Conformément aux exigences du projet, nous allons :
# 1.  Charger le dataset prétraité.
# 2.  Préparer les données pour l'entraînement et le test.
# 3.  Entraîner trois algorithmes de classification : K-Nearest Neighbors (KNN), Decision Tree (Arbre de Décision) et Naive Bayes.
# 4.  Évaluer la performance de chaque modèle à l'aide de métriques standard (accuracy, précision, rappel, F1-score, matrice de confusion, courbe ROC).
# 5.  Comparer les modèles pour déterminer le plus performant.
# 6.  Sauvegarder les modèles entraînés pour une utilisation future.
# 
# ---

# ### 1. Initialisation de l'Environnement et Chargement des Données Prétraitées
# 
# Cette section importe les bibliothèques Python nécessaires pour la modélisation, y compris les algorithmes de classification et les outils d'évaluation de Scikit-learn. Nous chargeons ensuite le dataset qui a été nettoyé et transformé lors de la phase de prétraitement.

# In[3]:


# Cellule 1: Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Pour la sauvegarde des modèles
import os # Pour créer le dossier models


# In[5]:


# Cellule 2: Chargement du dataset prétraité
# Le fichier CSV prétraité est situé dans le dossier 'data'
df_preprocessed = pd.read_csv('../data/telco_churn_preprocessed.csv')

print("--- Aperçu du dataset prétraité ---")
print(df_preprocessed.head())
print(f"\nDimensions du dataset prétraité : {df_preprocessed.shape}")


# #### Interprétation de la Cellule 2 : Chargement du Dataset Prétraité
# 
# Le dataset `telco_churn_preprocessed.csv` a été chargé avec succès. Il contient désormais `7043` lignes et un nombre de colonnes accru (par rapport au dataset original) en raison de l'encodage One-Hot des variables catégorielles. Toutes les colonnes sont numériques et prêtes à être utilisées pour l'entraînement des modèles de machine learning. L'absence de valeurs manquantes et le format numérique garantissent la compatibilité avec les algorithmes.

# ### 2. Séparation des Données en Ensembles d'Entraînement et de Test
# 
# Pour évaluer la performance de nos modèles de manière réaliste et éviter le surapprentissage, nous divisons le dataset en deux sous-ensembles : un ensemble d'entraînement (pour former le modèle) et un ensemble de test (pour évaluer ses performances sur des données non vues). Le professeur a précisé un split de 80% pour l'entraînement et 20% pour le test.

# In[9]:


# Cellule 3: Séparer les features (X) et la variable cible (y)
X = df_preprocessed.drop('Churn', axis=1)
y = df_preprocessed['Churn']

# Diviser les données en ensembles d'entraînement et de test
# random_state assure la reproductibilité du split
# stratify=y assure que la proportion des classes de Churn est la même dans train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dimensions de l'ensemble d'entraînement (X_train) : {X_train.shape}")
print(f"Dimensions de l'ensemble de test (X_test) : {X_test.shape}")
print(f"Distribution de la classe Churn dans y_train :\n{y_train.value_counts(normalize=True)}")
print(f"Distribution de la classe Churn dans y_test :\n{y_test.value_counts(normalize=True)}")


# #### Interprétation de la Cellule 3 : Split Train/Test
# 
# Le dataset a été correctement divisé en ensembles d'entraînement (80%) et de test (20%). L'utilisation de `stratify=y` a permis de maintenir la proportion des classes `No` et `Yes` de `Churn` identique dans les deux ensembles, ce qui est crucial étant donné le déséquilibre de classe observé lors de l'EDA. Cette étape garantit une évaluation non biaisée des modèles.

# ### 3. Entraînement et Évaluation des Modèles de Classification
# 
# Nous allons maintenant entraîner trois modèles de classification différents : K-Nearest Neighbors (KNN), Decision Tree (Arbre de Décision) et Naive Bayes. Pour chaque modèle, nous allons mesurer ses performances sur l'ensemble de test à l'aide de métriques clés telles que l'exactitude (accuracy), la précision, le rappel (recall) et le F1-score. Nous visualiserons également la matrice de confusion et la courbe ROC.

# In[13]:


# Cellule 4: Entraînement et évaluation du modèle KNN
print("\n--- Modèle 1 : K-Nearest Neighbors (KNN) ---")
# On choisit un k arbitraire pour ce TP, sans optimisation poussée
knn_model = KNeighborsClassifier(n_neighbors=5) # n_neighbors=5 est une valeur courante

knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_proba_knn = knn_model.predict_proba(X_test)[:, 1] # Probabilités de la classe positive

# Métriques d'évaluation
print(f"Accuracy (KNN): {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Precision (KNN): {precision_score(y_test, y_pred_knn):.4f}")
print(f"Recall (KNN): {recall_score(y_test, y_pred_knn):.4f}")
print(f"F1-Score (KNN): {f1_score(y_test, y_pred_knn):.4f}")
print("\nRapport de classification (KNN):\n", classification_report(y_test, y_pred_knn))

# Matrice de confusion
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Prédit Non-Churn', 'Prédit Churn'],
            yticklabels=['Réel Non-Churn', 'Réel Churn'])
plt.title('Matrice de Confusion (KNN)')
plt.xlabel('Prédiction')
plt.ylabel('Vraie Valeur')
plt.show()

# Courbe ROC
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)
plt.figure(figsize=(6, 5))
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Caractéristique de Fonctionnement du Récepteur (ROC) - KNN')
plt.legend(loc="lower right")
plt.show()


# #### Interprétation de la Cellule 4 : Modèle K-Nearest Neighbors (KNN)
# 
# Le modèle KNN a été entraîné avec un nombre de voisins de 5.
# - L'**Accuracy** indique la proportion globale de prédictions correctes.
# - La **Précision** est la capacité du modèle à ne pas prédire à tort la classe positive (minimiser les faux positifs).
# - Le **Rappel (Recall)** est la capacité du modèle à trouver tous les échantillons positifs (minimiser les faux négatifs).
# - Le **F1-Score** est la moyenne harmonique de la précision et du rappel, utile pour les datasets déséquilibrés.
# La **Matrice de Confusion** visualise les vrais positifs/négatifs et faux positifs/négatifs. La **Courbe ROC** et l'**AUC (Area Under the Curve)** mesurent la capacité du modèle à distinguer les classes. Un AUC proche de 1 indique une excellente séparation des classes. Les performances du KNN montrent un compromis entre précision et rappel, avec un AUC qui donne une idée de sa capacité générale à classer.

# In[16]:


# Cellule 5: Entraînement et évaluation du modèle Decision Tree
print("\n--- Modèle 2 : Decision Tree (Arbre de Décision) ---")
dt_model = DecisionTreeClassifier(random_state=42) # random_state pour la reproductibilité

dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
y_proba_dt = dt_model.predict_proba(X_test)[:, 1]

# Métriques d'évaluation
print(f"Accuracy (Decision Tree): {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision (Decision Tree): {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall (Decision Tree): {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score (Decision Tree): {f1_score(y_test, y_pred_dt):.4f}")
print("\nRapport de classification (Decision Tree):\n", classification_report(y_test, y_pred_dt))

# Matrice de confusion
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Prédit Non-Churn', 'Prédit Churn'],
            yticklabels=['Réel Non-Churn', 'Réel Churn'])
plt.title('Matrice de Confusion (Decision Tree)')
plt.xlabel('Prédiction')
plt.ylabel('Vraie Valeur')
plt.show()

# Courbe ROC
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)
plt.figure(figsize=(6, 5))
plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Caractéristique de Fonctionnement du Récepteur (ROC) - Decision Tree')
plt.legend(loc="lower right")
plt.show()


# #### Interprétation de la Cellule 5 : Modèle Decision Tree
# 
# Le modèle Decision Tree a été entraîné. Les arbres de décision sont des modèles intuitifs, mais ils peuvent être sujets au surapprentissage si leur profondeur n'est pas limitée. Les métriques et visualisations nous aideront à comprendre sa performance. Une différence notable par rapport à KNN pourrait être observée dans la manière dont il gère les faux positifs/négatifs. L'AUC fournira une comparaison globale de sa capacité de classification.

# In[19]:


# Cellule 6: Entraînement et évaluation du modèle Naive Bayes
print("\n--- Modèle 3 : Naive Bayes (GaussianNB) ---")
nb_model = GaussianNB() # GaussianNB pour des features numériques continues

nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
y_proba_nb = nb_model.predict_proba(X_test)[:, 1]

# Métriques d'évaluation
print(f"Accuracy (Naive Bayes): {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Precision (Naive Bayes): {precision_score(y_test, y_pred_nb):.4f}")
print(f"Recall (Naive Bayes): {recall_score(y_test, y_pred_nb):.4f}")
print(f"F1-Score (Naive Bayes): {f1_score(y_test, y_pred_nb):.4f}")
print("\nRapport de classification (Naive Bayes):\n", classification_report(y_test, y_pred_nb))

# Matrice de confusion
cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Prédit Non-Churn', 'Prédit Churn'],
            yticklabels=['Réel Non-Churn', 'Réel Churn'])
plt.title('Matrice de Confusion (Naive Bayes)')
plt.xlabel('Prédiction')
plt.ylabel('Vraie Valeur')
plt.show()

# Courbe ROC
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)
plt.figure(figsize=(6, 5))
plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_nb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Caractéristique de Fonctionnement du Récepteur (ROC) - Naive Bayes')
plt.legend(loc="lower right")
plt.show()


# #### Interprétation de la Cellule 6 : Modèle Naive Bayes
# 
# Le modèle Naive Bayes (plus spécifiquement Gaussian Naive Bayes, adapté aux données continues) a été entraîné. Ce modèle est basé sur le théorème de Bayes et l'hypothèse d'indépendance des caractéristiques, ce qui le rend rapide mais potentiellement moins précis si les hypothèses ne sont pas respectées. Ses métriques nous permettront de juger de son efficacité par rapport aux autres modèles pour ce dataset.

# ### 4. Comparaison des Modèles
# 
# Après avoir entraîné et évalué les trois modèles individuellement, il est essentiel de les comparer pour déterminer lequel offre la meilleure performance globale pour la tâche de prédiction du churn. Nous allons résumer leurs principales métriques et éventuellement tracer leurs courbes ROC sur un même graphique pour une comparaison visuelle.

# In[23]:


# Cellule 7: Comparaison des modèles
print("\n--- Comparaison des Modèles ---")

# Résumé des métriques clés
results = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree', 'Naive Bayes'],
    'Accuracy': [accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_nb)],
    'Precision': [precision_score(y_test, y_pred_knn), precision_score(y_test, y_pred_dt), precision_score(y_test, y_pred_nb)],
    'Recall': [recall_score(y_test, y_pred_knn), recall_score(y_test, y_pred_dt), recall_score(y_test, y_pred_nb)],
    'F1-Score': [f1_score(y_test, y_pred_knn), f1_score(y_test, y_pred_dt), f1_score(y_test, y_pred_nb)],
    'AUC': [roc_auc_knn, roc_auc_dt, roc_auc_nb]
})

print(results.sort_values(by='F1-Score', ascending=False)) # Trier par F1-Score, ou AUC

# Visualisation des courbes ROC comparées
plt.figure(figsize=(8, 7))
plt.plot(fpr_knn, tpr_knn, color='red', lw=2, label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_dt, tpr_dt, color='blue', lw=2, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot(fpr_nb, tpr_nb, color='green', lw=2, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Comparaison des Courbes ROC des Modèles')
plt.legend(loc="lower right")
plt.show()


# #### Interprétation de la Cellule 7 : Comparaison des Modèles
# 
# Le tableau récapitulatif des métriques et le graphique ROC comparatif permettent d'identifier le modèle le plus performant pour la prédiction du churn. Pour un dataset déséquilibré comme le nôtre, le **F1-Score** et l'**AUC (Area Under the Curve)** sont souvent des métriques plus révélatrices que la simple accuracy.
# - Le modèle avec le F1-Score le plus élevé est généralement celui qui trouve le meilleur équilibre entre précision et rappel pour la classe minoritaire.
# - L'AUC mesure la capacité globale du modèle à distinguer les classes positives et négatives. Plus l'AUC est proche de 1, meilleure est la performance du modèle.
# 
# (Insérer ici un commentaire sur le modèle le plus performant en fonction de vos résultats d'exécution).

# ### 5. Sauvegarde des Modèles Entraînés
# 
# Une fois que les modèles sont entraînés, il est essentiel de les sauvegarder pour pouvoir les réutiliser ultérieurement sans avoir à les ré-entraîner. Conformément aux exigences du projet, nous allons sauvegarder chaque modèle individuellement dans un nouveau répertoire `models/`.

# In[27]:


# Cellule 8: Créer le répertoire 'models' s'il n'existe pas
models_dir = '../models/'
os.makedirs(models_dir, exist_ok=True)

# Sauvegarder chaque modèle
joblib.dump(knn_model, os.path.join(models_dir, 'knn_model.pkl'))
joblib.dump(dt_model, os.path.join(models_dir, 'decision_tree_model.pkl'))
joblib.dump(nb_model, os.path.join(models_dir, 'naive_bayes_model.pkl'))

print(f"Modèles sauvegardés dans le répertoire : {models_dir}")
print("- knn_model.pkl")
print("- decision_tree_model.pkl")
print("- naive_bayes_model.pkl")


# #### Conclusion du Notebook de Modélisation et Évaluation
# 
# Ce notebook a complété la phase de modélisation du projet Telco Customer Churn. Nous avons :
# 1.  Chargé les données prétraitées issues du notebook précédent.
# 2.  Divisé le dataset en ensembles d'entraînement et de test avec un ratio 80/20 et stratification.
# 3.  Entraîné et évalué trois algorithmes de classification : K-Nearest Neighbors, Decision Tree et Naive Bayes.
# 4.  Comparé leurs performances à l'aide de métriques clés et de courbes ROC.
# 5.  Sauvegardé les modèles entraînés dans le dossier `models/` pour leur réutilisation ou déploiement futur.
# 
# Le modèle le plus performant pour ce cas d'étude a été identifié (mentionnez ici lequel, ex: "Le modèle Decision Tree a montré la meilleure performance globale en termes de F1-Score et d'AUC"). Ces modèles peuvent maintenant être utilisés pour prédire le churn de nouveaux clients.
# 
# ---

# In[ ]:




