# Prédiction du risque de défaut de crédit

Projet personnel de machine learning : prédire si un emprunteur fera défaut sur son prêt, à partir de ses caractéristiques et de celles du prêt.

## Données

Credit Risk Dataset (Kaggle) : https://www.kaggle.com/datasets/laotse/credit-risk-dataset

- 32 581 dossiers de crédit, 12 variables
- Cible : loan_status (0 = remboursé, 1 = défaut)
- Classes déséquilibrées : 78 % / 22 %

## Démarche

1. Exploration des données
2. Nettoyage (valeurs impossibles, doublons)
3. Préparation (train/test stratifié, imputation, encodage, normalisation)
4. Modélisation par régression logistique
5. Évaluation (matrice de confusion, précision, rappel, AUC)

Règle suivie à chaque étape : aucune information du jeu de test n'influence l'apprentissage (pas de fuite de données).

## Résultats

| Métrique | Valeur |
|---|---|
| Accuracy | 0,86 |
| Précision (défaut) | 0,76 |
| Rappel (défaut) | 0,56 |
| AUC | 0,87 |

L'accuracy de 0,86 est trompeuse : sur des données déséquilibrées, elle masque le fait que le modèle ne détecte que 56 % des défauts réels - précisément l'erreur la plus coûteuse pour un prêteur. L'AUC de 0,87 montre en revanche qu'il hiérarchise bien le risque.

## Stack

Python, pandas, NumPy, Matplotlib, scikit-learn
