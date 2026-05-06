# Modèle de prédiction maladie cardio-vasculaire à l'aide de régression logistique
Dans data/ un .txt est là comme support au .csv qui contient une liste de patient qui servira comme exemple au projet.

### Veille

- Certification Hébergeur de données de Santé (HDS) :
C'est la certification obligatoire pour pouvoir travailler avec des données de santé en France.
Elle est basée sur 4 principes, d'abord 2 principes de sécurité :
 - Sécurité physique : 
 Les centres de données doivent être sécurisés avec toutes les mesures de protection contre les évènements physiques naturels ou humains.

 - Cyber-Sécurité :
 Les données doivent être chiffrées, leurs accès en écriture et lecture contrôlées et toutes les mesures de cybersécurité doivent être mises en place.

 - Disponibilité :
 Les données doivent être accessibles à tout moment par les utilisateurs autorisés sans interruption.

 - RGPD :
 Les données doivent être conformes aux lois et règlements propres aux données personnelles.

- Régression Logistique :
C'est une technique d'analyse de données qui utilise les mathématiques pour trouver les relations entre des facteurs de données.
La ligne de régression est une courbe sigmoïde et est utilisée lorsqu'on veut un résultat binaire c'est à dire, oui ou non.

- Autre modèle de prédiction - Les arbres de décision :
Les arbres de décision sont des modèles très interprétables car ils sont basés sur des déclarations "if-then". Cela les rend facile à comprendre et à expliquer. Cependant, à chaque question, le modèle va optimiser au maximum, de manière trop "locale" et ne va pas généraliser. De plus, le modèle est hautement instable, le premier "split" est très sensible et le reste des feuilles de l'arbre suivent l'effet cascade.