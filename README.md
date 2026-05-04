# Prédiction des Risques Cardio-Vasculaires — Régression Logistique

## Contexte du projet

Les maladies cardiovasculaires sont la **deuxième cause de mortalité en France**, avec 300 000 à 400 000 accidents par an (AVC, infarctus du myocarde), dont un tiers sont mortels.

Ce projet s'inscrit dans une démarche de **médecine préventive**. L'objectif est de développer un outil de **diagnostic rapide du risque cardio-vasculaire** à partir de données cliniques et comportementales, en utilisant un algorithme de **classification binaire** (0 = sain, 1 = maladie cardiovasculaire).

---

## Veille — Régression Logistique

La **régression logistique** est un algorithme de classification supervisée qui prédit la probabilité qu'une observation appartienne à une classe binaire (0 ou 1).

**Principe mathématique** :

Au lieu de modéliser directement la classe, la régression logistique modélise la probabilité via la fonction **sigmoïde** :

$$P(y=1 | X) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{X} + b)}}$$

- Si P > 0.5 → classe 1 (malade)
- Si P ≤ 0.5 → classe 0 (sain)

**Apprentissage** : Les poids **w** et le biais **b** sont optimisés par descente de gradient sur la fonction de coût (entropie croisée binaire / *binary cross-entropy*) :

$$J(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

**Avantages** :
- Simple, rapide, interprétable (coefficients = importance des features)
- Fournit des probabilités (pas seulement une classe)
- Peu sujet au surapprentissage avec régularisation

**Limites** :
- Suppose une relation linéaire entre les features et le log-odds
- Ne capture pas les interactions non-linéaires complexes

---

## Les données

**Source** : Dataset `cardio_train.csv` — 70 000 patients.

| Colonne | Description | Type |
|---|---|---|
| `age` | Âge (converti de jours en années) | Numérique |
| `genre` | Genre (1 = Femme, 2 = Homme) | Catégoriel |
| `taille` | Taille en cm | Numérique |
| `poids` | Poids en kg | Numérique |
| `systolic` | Tension artérielle systolique (mmHg) | Numérique |
| `diastolic` | Tension artérielle diastolique (mmHg) | Numérique |
| `cholestrol` | Taux de cholestérol (1=normal, 2=élevé, 3=très élevé) | Catégoriel |
| `glucose` | Taux de glucose (1=normal, 2=élevé, 3=très élevé) | Catégoriel |
| `fumeur` | Tabagisme (0=non, 1=oui) | Binaire |
| `alcool` | Consommation d'alcool (0=non, 1=oui) | Binaire |
| `sportif` | Activité physique (0=non, 1=oui) | Binaire |
| `malade` | **Variable cible** — maladie cardiovasculaire (0=non, 1=oui) | Binaire |

### Analyse des données

**Pré-traitement appliqué** :
- Conversion de l'âge (jours → années)
- Suppression de la colonne `id` (non informative)
- Aucun doublon, aucune valeur manquante
- Suppression des **valeurs aberrantes médicalement impossibles** :
  - Tensions artérielles ≤ 0 ou physiologiquement extrêmes
  - Systolique ≤ diastolique (erreur de saisie)
  - Tailles hors [100–250] cm et poids hors [30–200] kg

**Observations clés** :
- Le dataset est **équilibré** (~50% sains / ~50% malades) — favorable pour la modélisation
- L'**âge** et la **tension systolique** sont les variables les plus corrélées au risque
- Le **cholestérol** et le **glucose** sont positivement corrélés (facteurs cliniques en aval)

---

## Algorithmes utilisés

### 1. Régression Logistique (Scikit-Learn)

Implémentation avec **optimisation des hyperparamètres par GridSearchCV** (validation croisée 5-fold, optimisée sur le recall) :

| Hyperparamètre | Valeurs testées | Description |
|---|---|---|
| `C` | 0.01, 0.1, 1, 10, 100 | Inverse du coefficient de régularisation |
| `solver` | lbfgs, liblinear | Algorithme d'optimisation |
| `max_iter` | 200, 500 | Nombre max d'itérations |

### 2. Régression Logistique from Scratch

Implémentation manuelle en Python pur (sans bibliothèque ML) :
- Descente de gradient (gradient descent)
- Fonction sigmoïde avec clipping pour éviter l'overflow
- Matrice de confusion calculée sans sklearn

---

## Évaluation des modèles

| Métrique | Sklearn (GridSearch) | Custom (scratch) |
|---|---|---|
| Accuracy | ~0.72 | ~0.65 |
| Recall | ~0.72 | ~0.62 |
| Precision | ~0.72 | ~0.66 |
| F1-score | ~0.72 | ~0.64 |
| AUC-ROC | ~0.78 | — |

**Métrique prioritaire : le Recall**

Dans un dépistage médical cardio-vasculaire, **manquer un patient malade (faux négatif) est plus dangereux** qu'alerter un patient sain (faux positif). Le recall (= TP / (TP + FN)) mesure directement cette capacité à ne pas manquer les malades.

---

## Conclusion

Le modèle sklearn optimisé obtient de meilleures performances que l'implémentation custom, grâce aux optimisations numériques et à la recherche d'hyperparamètres. Les deux approches convergent vers des résultats similaires, validant la logique de la régression logistique.

Les facteurs de risque identifiés par le modèle (âge, tension artérielle, cholestérol) sont cohérents avec les **12 facteurs de risque cardiovasculaire** reconnus par la littérature médicale. La régression logistique offre un bon compromis entre **interprétabilité** et **performance**, ce qui est essentiel dans un contexte médical où les décisions doivent être justifiables.