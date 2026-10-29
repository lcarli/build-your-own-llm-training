# Lab 1 — Tokeniseur : le texte devient des nombres

*Objectif : transformer du texte en une liste de nombres (IDs), puis revenir au texte.*

### Pourquoi
Un modèle de langage ne lit pas les lettres : il effectue uniquement des calculs avec des nombres. La première étape consiste donc à découper le texte en morceaux, appelés **tokens**, et à attribuer un nombre (un **ID**) à chacun. L'ensemble des tokens connus constitue le **vocabulaire**.

Ici, chaque caractère est un token. Les modèles comme GPT utilisent des morceaux plus grands, par exemple des parties de mots, mais l'idée reste la même.

### Prédisez
Combien de caractères différents y a-t-il dans "o rato roeu a roupa do rei de roma" ?
@answer **11** : espace, a, d, e, i, m, o, p, r, t, u.

### Copiez et collez
@code b01_tokenizer
- `sorted(set(text))` construit le vocabulaire : chaque caractère une seule fois, dans l'ordre.
- `stoi` (*string to index*) et `itos` (*index to string*) sont les deux dictionnaires de traduction : caractère → ID et ID → caractère.
- `encode` transforme le texte en IDs ; `decode` fait le chemin inverse.

### Vérifiez et observez
@verify
Dans le panneau :
- avec le texte de base "o rato roeu a roupa do rei de roma", le vocabulaire compte 11 caractères, "roma" devient `[8, 6, 5, 1]` et `decode` renvoie "roma" ;
- avec le corpus **{{corpus_name}}**, le vocabulaire compte **{{vocab_size}}** caractères. C'est ce tokeniseur que le reste de l'atelier utilise.

### Expérimentez
- Encodez "Roma", avec un R majuscule. Vous obtenez `KeyError: 'R'`, ce qui est normal : le tokeniseur ne connaît que les caractères qu'il a vus dans le texte de base. De nombreux LLM réels, comme GPT-2, partent d'octets et ne rencontrent donc jamais de caractère inconnu.
- Remplacez le texte de base par une phrase de votre choix et observez le vocabulaire changer.
- Dans le tableau du corpus, cherchez les caractères les plus fréquents et les moins fréquents.

### Pour réfléchir
À chaque étape, parmi combien de possibilités le modèle choisira-t-il le prochain token ?
@answer **{{vocab_size}}**, la taille du vocabulaire. Dans le Lab 4, ce nombre devient la dernière dimension de la sortie du modèle.
