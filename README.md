<p align="center">
  <a href="#fr">🇫🇷 Français</a> · <a href="#en">🇬🇧 English</a>
</p>

> ⚠️ **Projet pédagogique : ce modèle n'est pas un outil de diagnostic médical.**
> *Educational project: this model is not a medical diagnostic tool.*

---

<a id="fr"></a>

# Prédiction du risque cardio-vasculaire par régression logistique

**Estimer, à partir de quelques mesures de routine, si un patient présente un risque de maladie cardio-vasculaire, pour l'orienter plus tôt vers un dépistage.**

## En bref

- **Problème** : repérer rapidement les patients à risque cardio-vasculaire à partir de données simples (âge, tension, poids, cholestérol, habitudes de vie).
- **Données** : les dossiers de 70 000 patients (fichier fourni par l'école), dont 68 676 conservés après nettoyage des erreurs de saisie.
- **Méthode** : un modèle de classification (régression logistique) programmé deux fois, une fois « à la main » pour en comprendre le fonctionnement et une fois avec une bibliothèque standard réglée pour ne pas passer à côté des malades.
- **Résultat** : sur les patients malades du jeu de test, le modèle en repère plus de 8 sur 10.

## Contexte et problématique

Les maladies cardio-vasculaires sont la deuxième cause de mortalité en France, avec 300 000 à 400 000 accidents par an (AVC, infarctus), dont un tiers sont mortels. Le projet s'inscrit dans une démarche de médecine préventive.

**Question posée** : à partir des données cliniques et comportementales d'un patient, peut-on prédire s'il présente une maladie cardio-vasculaire (0 = sain, 1 = malade) ?

Le sujet imposait aussi d'implémenter la régression logistique sans bibliothèque de machine learning, de la comparer à la version scikit-learn, et de prédire le risque d'un patient fictif, « Arthur ».

**Veille réglementaire** : en France, travailler sur des données de santé impose un hébergeur certifié HDS (Hébergeur de Données de Santé). Cette certification repose sur la sécurité physique des centres de données, la cybersécurité (chiffrement, contrôle des accès), la disponibilité des données et la conformité au RGPD.

## Données

- **Source** : jeu de données fourni par LaPlateforme_ dans le cadre du projet (source privée école), `data/raw/cardio_train.csv`.
- **Taille** : 70 000 patients, 12 variables plus un identifiant. Aucun doublon, aucune valeur manquante.
- **Après nettoyage** : 68 676 patients (1 324 lignes aberrantes retirées, soit 1,89 %), sauvegardés dans `data/clean/cardio_train_clean.csv`.
- **Équilibre des classes** : 50,5 % de sains et 49,5 % de malades.

| Variable | Description | Type |
|---|---|---|
| `age` | Âge (converti de jours en années) | Numérique |
| `genre` | 1 = femme, 2 = homme | Catégorielle |
| `taille` / `poids` | Taille (cm), poids (kg) | Numérique |
| `systolic` / `diastolic` | Tension artérielle systolique / diastolique (mmHg) | Numérique |
| `cholestrol` / `glucose` | 1 = normal, 2 = élevé, 3 = très élevé | Catégorielle |
| `fumeur` / `alcool` / `sportif` | 0 = non, 1 = oui | Binaire |
| `malade` | **Cible** : maladie cardio-vasculaire (0 = non, 1 = oui) | Binaire |

Certains facteurs de risque cités dans le sujet (antécédents familiaux, sommeil, dépression) ne figurent pas dans ce jeu de données.

## Méthode

### 1. Nettoyage

- Conversion de l'âge de jours en années et suppression de la colonne `id`.
- Suppression des valeurs médicalement impossibles : tension systolique hors ]0 ; 250] mmHg, diastolique hors ]0 ; 200] mmHg, systolique inférieure ou égale à la diastolique, taille hors [100 ; 250] cm, poids hors [30 ; 200] kg. Le jeu brut contenait par exemple des tensions négatives (−150 mmHg) ou de 16 020 mmHg.

### 2. Exploration

Distributions, box plots et matrice de corrélation. L'âge et la tension systolique sont les variables les plus liées à la maladie, et le cholestérol et le glucose sont corrélés entre eux. Le notebook `exploration_kylliann.ipynb` approfondit le contrôle qualité (types, doublons, anomalies par tranche d'âge).

### 3. Modèles testés

| Modèle | Détails |
|---|---|
| **Régression logistique scikit-learn** | Deux variables ajoutées (IMC, pression différentielle = systolique − diastolique). Standardisation, GridSearchCV en validation croisée 5-fold sur `C`, `solver`, `max_iter` et `class_weight`, scoring F1. Meilleurs paramètres : `C=0.1`, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=300` |
| **Régression logistique from scratch** | Classe Python et NumPy : sigmoïde, descente de gradient (taux d'apprentissage de 0,01, 1 000 epochs), matrice de confusion recodée à la main. Mêmes variables que le modèle scikit-learn |
| **Variante (`modelisation.ipynb`)** | Pipeline alternatif : baseline, tuning optimisé sur le recall, version from scratch avec régularisation L2, et Random Forest en bonus |

### 4. Choix final et pourquoi

Le modèle retenu est la **régression logistique scikit-learn** avec `class_weight='balanced'` et un **seuil de décision abaissé à 0,387** (au lieu de 0,5). C'est le plus performant des deux implémentations sur toutes les métriques. La régression logistique reste aussi interprétable, ce qui compte en contexte médical où une décision doit pouvoir se justifier.

## Résultats

**En clair** : sur les 6 789 patients malades du jeu de test, le modèle en repère 5 621, soit plus de 8 sur 10. En contrepartie, environ une alerte sur trois concerne une personne saine.

**Pourquoi ce compromis ?** On a privilégié le recall (la part des malades détectés), car ne pas rater un malade compte plus qu'une fausse alerte. Une fausse alerte coûte un examen complémentaire, alors qu'un malade non détecté ne sera pas suivi.

Jeu de test : 13 736 patients (20 % du jeu nettoyé).

| Métrique | scikit-learn (retenu) | From scratch |
|---|---|---|
| Recall (classe malade) | **0,8280** | 0,8087 |
| Précision | 0,6729 | 0,6652 |
| F1-score | 0,7424 | 0,7300 |
| Accuracy | 0,7161 | 0,7037 |
| AUC-ROC | 0,7999 | non calculée |
| Seuil de décision | 0,387 | 0,39 |
| Vrais positifs / faux négatifs | 5 621 / 1 168 | 5 501 / 1 301 |
| Faux positifs / vrais négatifs | 2 732 / 4 215 | 2 769 / 4 165 |

F1 moyen en validation croisée (5-fold) du meilleur modèle : 0,7072.

**Cas Arthur** (53 ans, homme, fumeur, sportif, 175 cm, 85 kg, cholestérol élevé, glucose normal) : probabilité de maladie de **58,1 %**, donc classé « à risque ».

<details>
<summary>Résultats de la variante <code>modelisation.ipynb</code> (seuil 0,5, nettoyage légèrement différent : 68 610 patients)</summary>

| Modèle | Accuracy | Précision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistique baseline | 0,7289 | 0,7521 | 0,6741 | 0,7110 | 0,7919 |
| Logistique tunée (recall) | 0,7298 | 0,7493 | 0,6819 | 0,7140 | 0,7920 |
| Logistique from scratch | 0,7288 | 0,7518 | 0,6746 | 0,7111 | 0,7920 |
| Random Forest (bonus) | 0,7356 | 0,7537 | 0,6915 | 0,7213 | 0,8008 |

Variables les plus influentes, d'après les deux modèles : la tension systolique, puis l'âge, la tension diastolique et le cholestérol. Arthur est aussi classé à risque (probabilité de 0,5737 avec le modèle tuné).
</details>

## Limites et pistes d'amélioration

- **Fausses alertes** : environ un tiers des patients signalés sont sains. Le seuil peut être ajusté selon le coût réel d'un examen complémentaire.
- **Choix du seuil** : il a été optimisé directement sur le jeu de test, ce qui rend l'estimation un peu optimiste. Il vaudrait mieux le fixer sur un jeu de validation séparé.
- **Données** : un seul jeu de données, sans validation externe, sans antécédents familiaux ni historique médical.
- **Modèle linéaire** : la régression logistique capte mal les interactions complexes. Le Random Forest testé en bonus fait légèrement mieux, et un gradient boosting est à essayer.
- **Deux pipelines parallèles** : les notebooks n'utilisent pas exactement le même nettoyage. Il faudrait les unifier.
- Pistes supplémentaires : calibration des probabilités et suivi des performances dans le temps.

## Structure du repo

| Fichier | Rôle |
|---|---|
| `exploration.ipynb` | **Notebook principal** : nettoyage, exploration, régression logistique scikit-learn et from scratch, comparaison, cas Arthur |
| `exploration_kylliann.ipynb` | Exploration détaillée et contrôle qualité des données |
| `modelisation.ipynb` | Pipeline alternatif : baseline, tuning sur le recall, from scratch, Random Forest, variables contributives |
| `data/raw/cardio_train.csv` | Données brutes (70 000 patients, séparateur `;`) |
| `data/raw/cardiovasc_description.txt` | Dictionnaire des variables d'origine |
| `data/clean/cardio_train_clean.csv` | Données nettoyées (générées par `exploration.ipynb`) |
| `presentation_slides_plan.md` | Plan de la soutenance |
| `requirements.txt` | Dépendances Python |

## Installation et exécution

```bash
git clone https://github.com/axel-achart/cardio-vasc-risks.git
cd cardio-vasc-risks
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook exploration.ipynb
```

Les notebooks se lancent depuis la racine du repo (les chemins vers `data/` sont relatifs). Les sorties sont déjà enregistrées : il n'est pas nécessaire de les relancer pour consulter les résultats.

## Équipe

| Nom | GitHub |
|---|---|
| Axel Achart | [@axel-achart](https://github.com/axel-achart) |
| Hippolyte Geslain | [@Hippolyte-Geslain](https://github.com/Hippolyte-Geslain) |
| Kylliann Larcher | [@kylliann-larcher](https://github.com/kylliann-larcher) |

Projet réalisé à **LaPlateforme_** (Marseille), Bachelor IA/Data, de mars à juin 2026.

## Stack technique

Python 3.10 · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Jupyter

<p align="right"><a href="#en">English version ↓</a></p>

---

<a id="en"></a>

# Cardiovascular risk prediction with logistic regression

**Estimate, from a few routine measurements, whether a patient is at risk of cardiovascular disease, so they can be referred for screening earlier.**

## At a glance

- **Problem**: quickly flag patients at cardiovascular risk from simple data (age, blood pressure, weight, cholesterol, lifestyle).
- **Data**: records of 70,000 patients (dataset provided by the school), 68,676 kept after removing data-entry errors.
- **Method**: a classification model (logistic regression) built twice, once "by hand" to understand how it works and once with a standard library, tuned so as not to miss sick patients.
- **Result**: among the sick patients in the test set, the model flags more than 8 out of 10.

## Context and problem statement

Cardiovascular diseases are the second leading cause of death in France, with 300,000 to 400,000 events per year (strokes, heart attacks), one third of them fatal. The project takes a preventive-medicine approach.

**Question**: from a patient's clinical and lifestyle data, can we predict whether they have cardiovascular disease (0 = healthy, 1 = sick)?

The brief also required implementing logistic regression without any machine learning library, comparing it with the scikit-learn version, and predicting the risk of a fictional patient, "Arthur".

**Regulatory watch**: in France, working with health data requires an HDS-certified host (Hébergeur de Données de Santé). The certification covers the physical security of data centers, cybersecurity (encryption, access control), data availability and GDPR compliance.

## Data

- **Source**: dataset provided by LaPlateforme_ for the project (private school source), `data/raw/cardio_train.csv`.
- **Size**: 70,000 patients, 12 variables plus an ID. No duplicates, no missing values.
- **After cleaning**: 68,676 patients (1,324 outlier rows removed, i.e. 1.89%), saved to `data/clean/cardio_train_clean.csv`.
- **Class balance**: 50.5% healthy and 49.5% sick.

| Variable | Description | Type |
|---|---|---|
| `age` | Age (converted from days to years) | Numeric |
| `genre` | Sex: 1 = female, 2 = male | Categorical |
| `taille` / `poids` | Height (cm), weight (kg) | Numeric |
| `systolic` / `diastolic` | Systolic / diastolic blood pressure (mmHg) | Numeric |
| `cholestrol` / `glucose` | 1 = normal, 2 = above normal, 3 = well above normal | Categorical |
| `fumeur` / `alcool` / `sportif` | Smoker / alcohol / physically active: 0 = no, 1 = yes | Binary |
| `malade` | **Target**: cardiovascular disease (0 = no, 1 = yes) | Binary |

Some risk factors mentioned in the brief (family history, sleep, depression) are not available in this dataset.

## Method

### 1. Cleaning

- Age converted from days to years, and the `id` column dropped.
- Removal of medically impossible values: systolic pressure outside ]0, 250] mmHg, diastolic outside ]0, 200] mmHg, systolic lower than or equal to diastolic, height outside [100, 250] cm, weight outside [30, 200] kg. The raw data contained, for example, negative blood pressures (−150 mmHg) or values of 16,020 mmHg.

### 2. Exploration

Distributions, box plots and a correlation matrix. Age and systolic pressure are the variables most related to the disease, and cholesterol and glucose are correlated with each other. The `exploration_kylliann.ipynb` notebook goes deeper into data quality checks (types, duplicates, anomalies by age group).

### 3. Models tested

| Model | Details |
|---|---|
| **scikit-learn logistic regression** | Two added features (BMI, pulse pressure = systolic − diastolic). Standardization, 5-fold cross-validated GridSearchCV over `C`, `solver`, `max_iter` and `class_weight`, F1 scoring. Best parameters: `C=0.1`, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=300` |
| **From-scratch logistic regression** | Python and NumPy class: sigmoid, gradient descent (learning rate of 0.01, 1,000 epochs), hand-coded confusion matrix. Same features as the scikit-learn model |
| **Variant (`modelisation.ipynb`)** | Alternative pipeline: baseline, recall-optimized tuning, from-scratch version with L2 regularization, and a bonus Random Forest |

### 4. Final choice and why

The selected model is the **scikit-learn logistic regression** with `class_weight='balanced'` and a **decision threshold lowered to 0.387** (instead of 0.5). It beats the from-scratch implementation on every metric. Logistic regression also stays interpretable, which matters in a medical setting where a decision must be explainable.

## Results

**In plain words**: out of the 6,789 sick patients in the test set, the model flags 5,621, more than 8 out of 10. In exchange, about one alert in three concerns a healthy person.

**Why this trade-off?** We prioritized recall (the share of sick patients detected), because missing a sick patient matters more than a false alarm. A false alarm costs an extra medical check, while an undetected patient gets no follow-up.

Test set: 13,736 patients (20% of the cleaned data).

| Metric | scikit-learn (selected) | From scratch |
|---|---|---|
| Recall (sick class) | **0.8280** | 0.8087 |
| Precision | 0.6729 | 0.6652 |
| F1-score | 0.7424 | 0.7300 |
| Accuracy | 0.7161 | 0.7037 |
| ROC AUC | 0.7999 | not computed |
| Decision threshold | 0.387 | 0.39 |
| True positives / false negatives | 5,621 / 1,168 | 5,501 / 1,301 |
| False positives / true negatives | 2,732 / 4,215 | 2,769 / 4,165 |

Mean cross-validated F1 (5-fold) of the best model: 0.7072.

**Arthur's case** (53, male, smoker, physically active, 175 cm, 85 kg, high cholesterol, normal glucose): **58.1%** probability of disease, so classified "at risk".

<details>
<summary>Results of the <code>modelisation.ipynb</code> variant (0.5 threshold, slightly different cleaning: 68,610 patients)</summary>

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Baseline logistic | 0.7289 | 0.7521 | 0.6741 | 0.7110 | 0.7919 |
| Tuned logistic (recall) | 0.7298 | 0.7493 | 0.6819 | 0.7140 | 0.7920 |
| From-scratch logistic | 0.7288 | 0.7518 | 0.6746 | 0.7111 | 0.7920 |
| Random Forest (bonus) | 0.7356 | 0.7537 | 0.6915 | 0.7213 | 0.8008 |

Most influential variables according to both models: systolic pressure, then age, diastolic pressure and cholesterol. Arthur is also classified at risk (probability of 0.5737 with the tuned model).
</details>

## Limitations and next steps

- **False alarms**: about a third of flagged patients are healthy. The threshold can be adjusted to the real cost of an extra check.
- **Threshold selection**: it was optimized directly on the test set, which makes the estimate slightly optimistic. It should be set on a separate validation set.
- **Data**: a single dataset, with no external validation, family history or medical history.
- **Linear model**: logistic regression struggles with complex interactions. The bonus Random Forest does slightly better, and gradient boosting is worth trying.
- **Two parallel pipelines**: the notebooks do not use exactly the same cleaning. They should be unified.
- Further ideas: probability calibration and performance monitoring over time.

## Repository structure

| File | Role |
|---|---|
| `exploration.ipynb` | **Main notebook**: cleaning, exploration, scikit-learn and from-scratch logistic regression, comparison, Arthur's case |
| `exploration_kylliann.ipynb` | Detailed data exploration and quality checks |
| `modelisation.ipynb` | Alternative pipeline: baseline, recall tuning, from scratch, Random Forest, feature contributions |
| `data/raw/cardio_train.csv` | Raw data (70,000 patients, `;` separator) |
| `data/raw/cardiovasc_description.txt` | Original data dictionary |
| `data/clean/cardio_train_clean.csv` | Cleaned data (generated by `exploration.ipynb`) |
| `presentation_slides_plan.md` | Presentation outline |
| `requirements.txt` | Python dependencies |

## Installation and usage

```bash
git clone https://github.com/axel-achart/cardio-vasc-risks.git
cd cardio-vasc-risks
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook exploration.ipynb
```

Run the notebooks from the repository root (paths to `data/` are relative). Outputs are already saved, so there is no need to re-run them to see the results.

## Team

| Name | GitHub |
|---|---|
| Axel Achart | [@axel-achart](https://github.com/axel-achart) |
| Hippolyte Geslain | [@Hippolyte-Geslain](https://github.com/Hippolyte-Geslain) |
| Kylliann Larcher | [@kylliann-larcher](https://github.com/kylliann-larcher) |

Project carried out at **LaPlateforme_** (Marseille), Bachelor in AI/Data, March to June 2026.

## Tech stack

Python 3.10 · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Jupyter

<p align="right"><a href="#fr">Version française ↑</a></p>
