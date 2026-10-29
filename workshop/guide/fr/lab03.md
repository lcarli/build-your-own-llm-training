# Lab 3 — Embeddings : chaque token devient un vecteur

*Objectif : attribuer à chaque token une liste de nombres que le modèle peut ajuster.*

### Pourquoi
Un ID n'est qu'une étiquette : l'ID 8 n'est ni « plus grand » ni « semblable » à l'ID 7. Le modèle a besoin de nombres avec lesquels il peut effectuer des calculs et qui peuvent changer pendant l'entraînement.

La solution est un tableau : chaque token reçoit une ligne de **{{n_embd}} nombres**, son **embedding** (un vecteur). Au début, les nombres sont aléatoires. Pendant l'entraînement, ils sont ajustés pour améliorer les prédictions.

Ces nombres sont des **paramètres** : les valeurs que le modèle apprend. Ce tableau comporte {{vocab_size}} × {{n_embd}} = **{{params_embedding}}** paramètres.

### Prédisez
Avant l'entraînement, des lettres semblables, comme "a" et "e", auront-elles des vecteurs proches ?
@answer **Non.** Les vecteurs sont aléatoires au départ. Toute organisation qui apparaîtra ensuite proviendra de l'entraînement. Revenez à ce panneau après le Lab 7.

### Copiez et collez
@code b03_embeddings
- `nn.Module` est la base de tout composant de réseau neuronal dans PyTorch ; `super().__init__()` prépare cette base.
- `nn.Embedding(vocab_size, n_embd)` crée le tableau : une ligne par token, `n_embd` nombres par ligne.
- `forward` décrit ce qui se passe quand le composant est utilisé : chaque ID devient sa ligne du tableau, et la forme passe de `(B, T)` à `(B, T, n_embd)`.

### Vérifiez et observez
@verify
Dans le panneau :
- la carte de chaleur montre les vecteurs des caractères du texte saisi : une ligne par caractère, {{n_embd}} colonnes ;
- les caractères répétés ont des lignes identiques, car ils utilisent la même ligne du tableau ;
- le nuage de points projette les {{vocab_size}} vecteurs en 2 dimensions (PCA). Ce n'est qu'une ombre des {{n_embd}} nombres, mais elle aide à voir quels vecteurs sont proches les uns des autres.

### Expérimentez
- Saisissez un mot avec des lettres répétées, comme "arara", et repérez les lignes identiques.
- Observez la projection : les points sont-ils dispersés au hasard ?

### Pour réfléchir
GPT-2 possède un vocabulaire de 50.257 tokens. Avec des vecteurs de 128 nombres, combien de paramètres le seul tableau d'embeddings aurait-il ?
@answer 50.257 × 128 = **6.432.896** paramètres. C'est l'une des raisons pour lesquelles cet atelier utilise des caractères : avec {{vocab_size}} tokens et des vecteurs de {{n_embd}} nombres, notre tableau comporte {{params_embedding}} paramètres.
