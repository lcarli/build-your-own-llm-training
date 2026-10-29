# Lab 4 — Logits : la première prédiction

*Objectif : assembler le premier modèle complet, qui attribue un score à chaque prochain token possible.*

### Pourquoi
Chaque token possède déjà un vecteur. Nous devons maintenant transformer ce vecteur en **un score pour chacun des {{vocab_size}} tokens** qui peuvent suivre. Ces scores s'appellent des **logits** : plus le score est élevé, plus le modèle mise sur ce token.

C'est une couche linéaire (`nn.Linear`) qui s'en charge : {{n_embd}} nombres en entrée, {{vocab_size}} scores en sortie.

En réunissant l'embedding (Lab 3) et la couche linéaire, nous obtenons un modèle **bigramme** : il prédit le prochain token en regardant **uniquement le token actuel**.

### Prédisez
Un logit peut-il être négatif ?
@answer **Oui.** Un logit est un score, pas une probabilité. Dans le Lab 5, la softmax transforme les scores en probabilités.

### Copiez et collez
@code b04_bigram
- `from bricks.b03_embeddings import TokenEmbedding` réutilise le composant du Lab 3 : les LEGO en action.
- `self.lm_head` est la couche linéaire qui transforme chaque vecteur en {{vocab_size}} scores.
- Dans `forward`, les IDs deviennent des vecteurs et les vecteurs deviennent des logits : `(B, T)` → `(B, T, {{n_embd}})` → `(B, T, {{vocab_size}})`.

### Vérifiez et observez
@verify
Dans le panneau :
- la succession des formes : `(1, T)` → `(1, T, {{n_embd}})` → `(1, T, {{vocab_size}})` ;
- les scores pour le caractère qui suivrait le dernier caractère saisi ;
- le tableau {{vocab_size}} × {{vocab_size}} : chaque ligne est un caractère actuel, chaque colonne un prochain caractère possible. Sans entraînement, le motif est aléatoire.

Le modèle entier comporte **{{params_bigram}}** paramètres.

### Expérimentez
- Modifiez le dernier caractère du texte et observez les scores changer.
- Trouvez dans le tableau la ligne du caractère que vous avez saisi.

### Pour réfléchir
Que ne peut **pas** savoir ce modèle sur le texte ?
@answer Tout ce qui est venu **avant** le caractère actuel. Après "qu" ou un "u" seul, il donne exactement les mêmes scores. Résoudre ce problème est le rôle de l'attention, à partir du Lab 8.
