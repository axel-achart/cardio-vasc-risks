# cardio-vasc-risks

Projet de classification binaire pour estimer le risque cardio-vasculaire (0 = pas de risque, 1 = risque) a partir de donnees patients.

## Objectif

Construire et comparer des modeles de regression logistique pour aider au depistage rapide du risque cardio-vasculaire, avec une lecture metier des erreurs (faux negatifs/faux positifs).

## Donnees

- Source principale: `data/raw/cardio_train.csv`
- Taille brute: 70 000 lignes
- Cible: `cardio`
- Variables principales:
  - non modifiables disponibles dans ce dataset: age, sexe
  - mode de vie: smoke, alco, active
  - cliniques: ap_hi, ap_lo, cholesterol, gluc
  - anthropometrie: height, weight

Note: certains facteurs du brief (antecedents familiaux, sommeil, depression) ne sont pas presents dans ce jeu de donnees.

## Structure du repo

- `exploration.ipynb`: EDA, controle qualite, nettoyage et interpretations visuelles
- `modelisation.ipynb`: regression logistique sklearn, tuning, implementation from scratch, evaluation et cas Arthur
- `data/raw/`: donnees brutes et description

## Veille synthétique: regression logistique

La regression logistique est un modele de classification supervisee qui estime:

- un score lineaire sur les variables d'entree
- une probabilite via la sigmoide
- une classe finale selon un seuil (par defaut 0.5)

Forces:

- interpretable
- rapide a entrainer
- bonne baseline sur donnees tabulaires

Limites:

- relation lineaire sur les log-odds
- sensible aux outliers et au pretraitement
- depend du choix de seuil en production

## Pretraitement applique

Etapes appliquees de facon reproductible:

1. Suppression des doublons
2. Filtrage des valeurs aberrantes medicalement incoherentes:
	- `ap_hi <= ap_lo`
	- `ap_hi` hors [70, 250]
	- `ap_lo` hors [40, 150]
	- `height` hors [120, 220]
	- `weight` hors [30, 250]
3. Feature engineering:
	- `age_years = age / 365.25`
	- `bmi = weight / height^2`
4. Split stratifie train/test (80/20)

Impact du nettoyage:

- avant: 70 000 lignes
- apres: 68 610 lignes
- lignes retirees: 1 390 (1.99%)

## Modeles entraines

### 1) Baseline sklearn

Pipeline:

- `StandardScaler`
- `LogisticRegression`

Resultats test:

- Accuracy: 0.7289
- Precision: 0.7521
- Recall: 0.6741
- F1: 0.7110
- ROC-AUC: 0.7919

### 2) Modele sklearn tune (GridSearchCV)

Optimisation sur le recall (classe a risque), avec recherche sur:

- `C`
- `solver`
- `class_weight`

Meilleurs hyperparametres:

- `C=0.01`
- `class_weight='balanced'`
- `solver='lbfgs'`

Resultats test:

- Accuracy: 0.7298
- Precision: 0.7493
- Recall: 0.6819
- F1: 0.7140
- ROC-AUC: 0.7920

### 3) Regression logistique from scratch

Implementation manuelle (sigmoide + descente de gradient + regularisation L2).

Resultats test:

- Accuracy: 0.7288
- Precision: 0.7518
- Recall: 0.6746
- F1: 0.7111
- ROC-AUC: 0.7920

## Interpretation metier des metriques

Pour ce cas d'usage prevention:

- la metrique prioritaire est le recall de la classe 1
- objectif: minimiser les faux negatifs (patients a risque non detectes)

Comparaison erreurs critiques baseline vs tuned:

- Baseline: FN=2212, taux FN=32.59%
- Tuned: FN=2159, taux FN=31.81%

Le tuned reduit les faux negatifs, avec une legere hausse des faux positifs. Ce compromis est pertinent en prevention cardio-vasculaire.

## Cas Arthur (demande du brief)

Profil:

- 53 ans
- homme
- fumeur
- sportif
- 175 cm, 85 kg
- cholesterol au-dessus de la normale
- glucose normal
- systolique = moyenne du dataset nettoye
- diastolique = moyenne du 3e quartile (50%-75%) du dataset nettoye

Predictions:

- Modele tuned sklearn: proba risque = 0.5737, prediction = 1
- Modele scratch: proba risque = 0.5692, prediction = 1

Interpretation: Arthur est classe "a risque" par les deux modeles.

## Conclusion

Le projet repond au besoin de depistage rapide avec une regression logistique interpretable.

- Le pretraitement medicalement coherent ameliore la robustesse des predictions.
- Le modele tune est le meilleur choix operationnel pour la prevention, car il detecte davantage de cas a risque (recall superieur).
- Le modele from scratch valide la logique mathematique du modele et produit des performances proches de sklearn.

## Pistes d'amelioration

- Ajuster le seuil de decision selon le cout clinique FN/FP
- Ajouter une calibration des probabilites
- Comparer avec un autre algorithme (ex: Random Forest, XGBoost) en bonus
- Mettre en place un suivi des performances dans le temps
