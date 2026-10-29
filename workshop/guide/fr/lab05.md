# Lab 5 — Softmax, température et génération de texte

*Objectif : transformer des scores en probabilités et générer du texte, un token à la fois.*

### Pourquoi
Les logits sont des scores non normalisés. La **softmax** les transforme en **probabilités** : toutes positives et dont la somme vaut 1.

La **température** ajuste l'audace du modèle : les logits sont divisés par elle avant la softmax.
- Température basse (inférieure à 1) : le favori devient encore plus probable et le texte devient répétitif.
- Température élevée (supérieure à 1) : les chances se rapprochent et le texte devient plus imprévisible.

Générer du texte consiste à répéter trois étapes : calculer les probabilités du prochain token, **tirer au hasard** un token et l'ajouter à la fin. Puis recommencer.

### Prédisez
Que deviennent les probabilités lorsque la température se rapproche de 0 ?
@answer Presque toute la probabilité va au token ayant le score le plus élevé. À la limite, le modèle choisit toujours le favori.

### Copiez et collez
@code b05_sampling
- `logits[:, -1, :]` ne prend que la dernière position : c'est elle qui prédit le prochain token.
- `F.softmax(..., dim=-1)` transforme chaque ligne de scores en probabilités.
- `torch.multinomial` tire un token au hasard en respectant les probabilités.
- `torch.cat` ajoute le token tiré à la fin de la séquence.
- `@torch.no_grad()` indique à PyTorch que nous n'allons pas entraîner le modèle ici, ce qui économise de la mémoire et du temps.

### Vérifiez et observez
@verify
Dans le panneau :
- les 10 probabilités les plus élevées pour le prochain caractère et la somme de toutes les probabilités : 1 ;
- un texte de 200 caractères généré par **votre** `generate` ;
- les premières étapes de génération : le caractère tiré, sa probabilité et les autres possibilités.

### Expérimentez
- Déplacez la température à 0,1, puis à 1 et à 2, et observez les barres.
- Générez deux fois avec la même seed : le texte est identique. Changez la seed et comparez.

### Pour réfléchir
Le texte généré ressemble à une soupe de lettres. Pourquoi ?
@answer Parce que le modèle **n'a pas encore été entraîné** : les scores sont aléatoires. Dans le Lab 6, nous mesurons l'ampleur de l'erreur, et dans le Lab 7, le modèle apprend.
