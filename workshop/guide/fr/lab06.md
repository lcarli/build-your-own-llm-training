# Lab 6 — Loss : mesurer l'erreur

*Objectif : calculer un nombre unique qui indique à quel point le modèle se trompe.*

### Pourquoi
Pour apprendre, le modèle doit savoir à quel point il s'est trompé. La mesure utilisée par les LLM est la **cross-entropy** (entropie croisée) :

1. À chaque position, regardez la probabilité que le modèle a attribuée au token **réel** qui suivait.
2. Calculez −log de cette probabilité : c'est la « surprise » du modèle.
3. Faites la moyenne sur toutes les positions.

Si le modèle a attribué une probabilité de 1 au bon token, la surprise vaut 0. S'il a attribué une probabilité presque nulle, la surprise est énorme. Cette moyenne est la **loss** : plus elle est basse, mieux c'est.

### Prédisez
Quelle est la loss d'un modèle qui attribue la même probabilité à tous les {{vocab_size}} tokens ?
@answer **ln({{vocab_size}}) ≈ {{ln_vocab}}.** Chaque token reçoit 1/{{vocab_size}}, et −log(1/{{vocab_size}}) = ln({{vocab_size}}). C'est la loss d'un modèle qui « ne sait rien ».

### Copiez et collez
@code b06_loss
- `logits.shape` renvoie les trois dimensions : B extraits, T positions et V tokens possibles.
- `.view(B * T, V)` rassemble toutes les positions dans une seule liste de prédictions ; `targets.view(B * T)` fait de même avec les bonnes réponses.
- `F.cross_entropy` applique la softmax, calcule −log de la probabilité du bon token et fait la moyenne, le tout en une seule opération.

### Vérifiez et observez
@verify
Dans le panneau :
- la surprise du modèle à chaque position du texte saisi (des couleurs plus intenses indiquent davantage de surprise) et la probabilité qu'il a attribuée au caractère réel ;
- la loss moyenne, comparée à {{ln_vocab}}, la loss d'un modèle qui répartit la probabilité équitablement ;
- sans entraînement, la loss se situe généralement un peu au-dessus de {{ln_vocab}} : les scores aléatoires favorisent certains tokens sans raison.

### Expérimentez
- Comparez un texte ordinaire à une suite aléatoire de lettres. Avant l'entraînement, les deux ont une loss semblable.
- Revenez ici après le Lab 7 et répétez la comparaison.

### Pour réfléchir
Si le modèle a toujours raison, avec une probabilité de 1, quelle est la loss ?
@answer **0**, car −log(1) = 0. En pratique, la loss n'atteint jamais 0 avec du texte réel : le prochain caractère n'est jamais totalement prévisible.
