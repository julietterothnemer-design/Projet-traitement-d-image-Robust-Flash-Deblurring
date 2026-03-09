# Projet-traitement-d-image-Robust-Flash-Deblurring
Dans le cadre d'un projet de traitement d'image à l'Institut d'Optique, nous avons développé un algorithme basé sur le papier de recherche " Robust Flash Deblurring, Shaojie Zhuo, Dong Guo, Terence Sim School of Computing, National University of Singapore" permettant de reconstruire une image avec du flou de bougé à l'aide d'un flash


# Robust Flash Deblurring – Projet de traitement d'image

Ce projet consiste à reproduire la méthode de **Robust Flash Deblurring** proposée dans l'article de Raskar et al.

L'objectif est de reconstruire une image nette à partir de deux photographies de la même scène :

- une **image ambiante floue** \(B\)
- une **image avec flash plus nette** \(F\)

L'image flash contient des **contours plus nets**, tandis que l'image ambiante conserve **l'illumination naturelle de la scène**. La méthode combine ces deux sources d'information pour estimer le flou de mouvement et reconstruire une image plus nette.

---

# Principe de la méthode

La méthode est implémentée en **deux étapes principales**.

## 1. Estimation du noyau de flou

La première étape consiste à estimer le **noyau de flou \(K\)** à partir d'un patch de l'image.

Principe :
- sélection d'une zone avec des structures fortes (contours)
- alignement local des patchs flash et flou
- estimation du noyau via une optimisation **IRLS**
- contraintes appliquées sur le noyau :
  - positivité
  - normalisation
  - parcimonie
  - recentrage

Le noyau estimé est sauvegardé sous forme de fichier :
K_est_crop.npy



---

## 2. Reconstruction de l'image complète

Une fois le noyau estimé, la reconstruction de l'image complète est effectuée en minimisant :

\[
||I*K - B||^2 + \lambda_f M \rho(\nabla I - \nabla F) + \lambda_i |\nabla I|^\alpha
\]

avec :

- **I** : image reconstruite  
- **K** : noyau de flou estimé  
- **B** : image floue  
- **F** : image flash  
- **M** : masque de confiance  

La reconstruction utilise :

- une optimisation **IRLS**
- un solveur **Conjugate Gradient**
- une **contrainte de gradients guidée par le flash**
- une **régularisation de parcimonie des gradients**

La reconstruction est appliquée **canal par canal (RGB)**.

---

# Structure du projet



