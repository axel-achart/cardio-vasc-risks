# Plan de Slides - Projet cardio-vasc-risks

## Slide 1 - Titre et contexte
- Problématique: dépistage rapide du risque cardio-vasculaire
- Objectif: classifier 0 (pas de risque) / 1 (risque)

## Slide 2 - Données
- Source et taille du dataset
- Variables disponibles (âge, sexe, mode de vie, indicateurs cliniques)
- Limites du dataset (variables absentes du brief)

## Slide 3 - Démarche globale
- Exploration
- Nettoyage
- Modélisation sklearn
- Modélisation from scratch
- Évaluation et comparaison

## Slide 4 - Contrôle qualité des données
- Valeurs manquantes
- Doublons
- Contrôle des modalités

## Slide 5 - Valeurs aberrantes
- Règles médicales utilisées
- Nombre de lignes impactées
- Exemples concrets d'incohérences

## Slide 6 - Impact du nettoyage
- Lignes avant/après
- Interprétation métier de l'impact
- Dataset final prêt à modéliser

## Slide 7 - Baseline régression logistique
- Pipeline (standardisation + logistic regression)
- Métriques baseline
- Matrice de confusion

## Slide 8 - Tuning
- Hyperparamètres testés
- Meilleur modèle obtenu
- Comparaison baseline vs tuned

## Slide 9 - Modèle from scratch
- Principe d'implémentation (sigmoïde + gradient descent)
- Courbe de convergence
- Performance vs sklearn

## Slide 10 - Bonus Random Forest
- Résultats du bonus
- Forces/faiblesses vs logistic regression
- Cas où RF peut aider

## Slide 11 - Cas Arthur
- Hypothèses utilisées
- Probabilité prédite
- Classe finale
- Limites de l'interprétation

## Slide 12 - Conclusion
- Métrique prioritaire: recall classe 1
- Choix du meilleur compromis
- Perspectives: calibration, seuil, monitoring, données complémentaires
