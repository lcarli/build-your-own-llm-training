# Lab 0 — Préparation : le déroulement de l'atelier

*Objectif : vérifier que votre ordinateur est prêt et pratiquer le cycle de tous les laboratoires : lire, copier, coller et vérifier.*

### Pourquoi
Dans cet atelier, vous allez assembler un modèle de langage pièce par pièce, comme des LEGO. Chaque laboratoire suit le même cycle :

1. Lisez l'explication.
2. Cliquez sur **Copy** dans le bloc de code.
3. Dans VS Code, ouvrez le fichier indiqué et collez **sous** la ligne `# ---- PASTE BELOW ----`. Le fichier est enregistré automatiquement.
4. Revenez sur cette page et cliquez sur **Verify**.
5. La visualisation à droite se débloque et affiche toujours les chiffres réels de **votre** modèle.

Tout fonctionne sur votre ordinateur : sans cloud, sans GPU et sans compte payant.

### Prédisez
Combien d'opérations sur des nombres décimaux (multiplications et additions) votre ordinateur effectue-t-il par seconde ? Mille ? Un million ? Un milliard ?
@answer La vérification le mesure sur votre ordinateur en multipliant des tableaux de nombres. Le résultat apparaît en **GFLOP/s** : des milliards d'opérations par seconde. Comparez avec la personne à côté de vous.

### Copiez et collez
Ce premier code est un simple échauffement : une fonction qui renvoie une salutation.
@code b00_warmup
- `def` crée une fonction ; `name` est la valeur qu'elle reçoit.
- Le texte entre triples guillemets décrit ce que fait la fonction.
- `f"..."` construit un texte contenant la valeur de `name`.

### Vérifiez et observez
@verify
La vérification contrôle :
- la version de Python (3.10 ou plus récente) ;
- si PyTorch est installé et combien de cœurs du processeur il utilisera ;
- le texte d'entraînement : **{{corpus_name}}**, avec {{corpus_chars}} caractères ;
- si le dossier `bricks/` autorise l'écriture ;
- votre fonction `greet` ;
- la vitesse de votre ordinateur pour multiplier des matrices.

### Expérimentez
Faites volontairement des erreurs pour découvrir les messages d'aide :
1. Dans `bricks/b00_warmup.py`, supprimez les espaces au début de la deuxième ligne de code (celle qui commence par `"""`) et cliquez sur **Verify**. Le message indique la ligne et explique le problème d'indentation.
2. Annulez (Ctrl+Z). Remplacez maintenant les guillemets droits `"` de la ligne du `return` par des guillemets courbes `“` et `”`, comme ceux qu'insèrent Word et PowerPoint. Vérifiez à nouveau.
3. Corrigez le fichier ou cliquez sur **Complete this step** pour restaurer le code de référence.

> Copiez toujours le code avec le bouton **Copy** de ce guide. Les diapositives, PDF, e-mails et chats remplacent les guillemets et les espaces par des caractères que Python n'accepte pas.

### Pour réfléchir
Pourquoi cet atelier fait-il tout fonctionner sur votre ordinateur plutôt que d'utiliser un service cloud ?
@answer Pour que chaque chiffre à l'écran provienne de **votre** modèle, sans frais et sans dépendre d'Internet. Et pour montrer que le mécanisme d'un LLM tient sur un ordinateur portable : dans les grands modèles, c'est principalement l'échelle qui change (plus de données, plus de paramètres et beaucoup plus de calcul).
