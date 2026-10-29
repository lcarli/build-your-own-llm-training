# Lab 7 — Entraînement : le modèle apprend

*Objectif : répéter des milliers de fois le cycle qui ajuste les paramètres pour réduire la loss.*

### Pourquoi
Jusqu'ici, les {{params_bigram}} paramètres du modèle sont aléatoires. Entraîner le modèle consiste à les ajuster progressivement pour faire baisser la loss. Chaque **étape d'entraînement** fait cinq choses :

1. **Prédire :** le modèle calcule les logits d'un batch.
2. **Mesurer l'erreur :** calculer la loss (Lab 6).
3. **Effacer :** remettre à zéro les gradients de l'étape précédente.
4. **Calculer les gradients** (`backward`) : pour chaque paramètre, déterminer dans quelle direction le déplacer pour faire baisser la loss. PyTorch effectue ce calcul automatiquement.
5. **Mettre à jour :** l'optimiseur (AdamW) déplace chaque paramètre un peu dans cette direction.

La taille de ce « peu » est la **learning rate** (taux d'apprentissage).

### Prédisez
Une learning rate 10 fois plus grande fait-elle apprendre le modèle 10 fois plus vite ?
@answer **Pas toujours.** De grands pas peuvent dépasser la cible et faire osciller la loss ; des pas trop petits ralentissent l'apprentissage. Testez dans le panneau.

### Copiez et collez
@code b07_training
- `make_optimizer` crée AdamW, qui ajustera tous les paramètres du modèle.
- `train_step` est une étape complète : prédire, mesurer l'erreur, effacer, calculer les gradients et mettre à jour.
- `loss.item()` renvoie la loss sous forme d'un nombre ordinaire pour tracer le graphique.

### Vérifiez et observez
@verify
Après la vérification, cliquez sur **Start** dans le panneau et suivez :
- les courbes de loss d'**entraînement** et de **validation**, mises à jour tous les 250 pas ;
- la ligne en pointillés : la **limite du bigramme**, {{bigram_floor}}. C'est la plus petite loss d'entraînement possible pour un modèle qui ne regarde que le caractère actuel, calculée en comptant les paires de caractères du texte d'entraînement. Notre modèle peut rester un peu au-dessus ;
- la **galerie de l'évolution** : la même demande de texte, avec la même seed, à différents moments de l'entraînement ;
- le tableau {{vocab_size}} × {{vocab_size}} qui se structure.

### Expérimentez
- Entraînez avec des learning rates de 0,1, 0,01 et 0,001 et comparez les courbes. Entre deux entraînements, cliquez sur **Reset model** : sinon, **Start** continue à entraîner le même modèle. Les courbes précédentes restent sur le graphique sous forme de lignes fines.
- Après l'entraînement, revenez aux Labs 3, 4, 5 et 6 : les panneaux montrent maintenant le modèle entraîné. Les vecteurs se sont-ils organisés ? Les probabilités sont-elles cohérentes ? La loss a-t-elle baissé ?

### Pour réfléchir
La loss s'est stabilisée près de la limite du bigramme. Que manque-t-il pour que le modèle écrive mieux ?
@answer **Du contexte.** En ne regardant qu'un seul caractère, aucun ajustement des paramètres ne permet de dépasser cette limite. L'étape suivante consiste à laisser chaque token regarder les précédents : la **self-attention**, à partir du Lab 8.
