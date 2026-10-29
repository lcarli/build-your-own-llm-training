# Lab 2 — Exemples d'entraînement : un texte, des milliers de questions

*Objectif : transformer le corpus en paires « contexte → prochain token » et séparer entraînement et validation.*

### Pourquoi
Le modèle apprend une seule tâche : **prédire le prochain token**. Le texte lui-même fournit déjà les bonnes réponses. Dans une fenêtre de 8 caractères, le modèle s'exerce à faire 8 prédictions : avec le 1er caractère, prédire le 2e ; avec les 2 premiers, prédire le 3e ; et ainsi de suite, jusqu'à prédire le caractère qui suit immédiatement la fenêtre.

Le texte est également divisé en deux parties :
- **entraînement** (90%) : le modèle apprend à partir de cette partie ;
- **validation** (10%) : le modèle ne s'entraîne jamais dessus ; elle sert à mesurer s'il a appris quelque chose qui s'applique à du texte nouveau.

Pour l'entraînement, nous tirons au hasard plusieurs extraits à la fois. Ce groupe d'extraits est un **batch**.

### Prédisez
Une fenêtre `x` contient 8 tokens. Combien d'exemples d'entraînement (contexte → cible) contient-elle ?
@answer **8**. La cible `y` est la même fenêtre décalée d'une position vers l'avant : chaque position de `x` trouve sa réponse à la même position dans `y`.

### Copiez et collez
@code b02_batches
- `torch.tensor` transforme la liste d'IDs en un **tenseur**, la structure de nombres de PyTorch.
- `split_train_val` découpe le tenseur en deux parties : 90% et 10%.
- `get_batch` tire au hasard `batch_size` positions de départ et découpe deux fenêtres à partir de chacune : `x`, le contexte, et `y`, la même fenêtre une position plus loin.
- `torch.stack` empile les fenêtres dans un tableau de forme `(B, T)` : B extraits de T tokens chacun.

### Vérifiez et observez
@verify
Dans le panneau :
- l'entraînement contient **{{train_tokens}}** tokens et la validation, **{{val_tokens}}** ;
- un batch de {{batch_size_demo}} extraits de {{block_size_demo}} tokens : comparez `x` et `y` et repérez le décalage d'une position ;
- les {{block_size_demo}} paires « contexte → cible » de la première ligne.

### Expérimentez
- Cliquez sur **New batch** pour tirer d'autres extraits au hasard.
- Modifiez la taille de la fenêtre (*block size*) et recomptez les paires.

### Pour réfléchir
Pourquoi réserver 10% du texte que le modèle ne voit jamais pendant l'entraînement ?
@answer Parce qu'un modèle peut **mémoriser** le texte d'entraînement. La loss de validation montre s'il a appris des régularités qui fonctionnent sur du texte nouveau. Les deux courbes apparaissent côte à côte dans le Lab 7.
