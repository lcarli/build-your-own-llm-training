# Build Your Own LLM — Laboratoire interactif

[Português](readme-pt.md) | [English](readme-en.md) | [Français](readme-fr.md)

**Convention du projet :** les commentaires, docstrings et messages du code doivent toujours être rédigés en anglais. L'interface utilise l'anglais ; les guides et les textes d'entraînement conservent leurs langues.

Un atelier guidé où chaque étudiant assemble, pièce par pièce, un **modèle de langage à caractères** (un bigramme dans cette version ; le mini-GPT viendra dans les prochains laboratoires). Tout fonctionne **sur l'ordinateur local, avec le CPU** : sans cloud, sans GPU, sans services payants et sans GitHub Copilot.

Dans chaque laboratoire, une page locale affiche un guide avec des explications et du code prêt à utiliser. L'étudiant **copie** le code, le **colle** dans le fichier indiqué dans VS Code, puis clique sur **Verify** (Vérifier). La page débloque ensuite une visualisation avec les valeurs **réelles** de son modèle : aucune donnée inventée.

| Lab | Thème | Fichier de l'étudiant |
|---|---|---|
| 0 | Préparation : le déroulement de l'atelier | `bricks/b00_warmup.py` |
| 1 | Tokeniseur : transformer le texte en nombres | `bricks/b01_tokenizer.py` |
| 2 | Exemples d'entraînement (batches) | `bricks/b02_batches.py` |
| 3 | Embeddings | `bricks/b03_embeddings.py` |
| 4 | Logits : la première prédiction | `bricks/b04_bigram.py` |
| 5 | Softmax, température et génération | `bricks/b05_sampling.py` |
| 6 | Loss : mesurer l'erreur | `bricks/b06_loss.py` |
| 7 | Entraînement | `bricks/b07_training.py` |

## Prérequis

- **Python 3.10 ou plus récent** (Windows, macOS ou Linux).
- **PyTorch** (version CPU). C'est la seule dépendance : le serveur utilise uniquement la bibliothèque standard de Python et la page est en HTML/CSS/JS pur, sans bibliothèques externes.
- Un navigateur récent (Chrome, Edge, Firefox ou Safari) et **VS Code**.
- N'importe quel ordinateur portable ordinaire : le modèle ne compte que quelques milliers de paramètres et s'entraîne en quelques secondes.

## Préparation (formateur)

1. **Créez un environnement virtuel** dans le dossier du projet :

   | Système | Commandes |
   |---|---|
   | Windows (PowerShell) | `py -3 -m venv .venv` et `.venv\Scripts\Activate.ps1` |
   | Windows (cmd) | `py -3 -m venv .venv` et `.venv\Scripts\activate.bat` |
   | macOS / Linux | `python3 -m venv .venv` et `source .venv/bin/activate` |

   Si PowerShell bloque le script d'activation : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

2. **Installez PyTorch pour CPU** (téléchargement bien plus petit que la version GPU) :

   ```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Préparez le texte d'entraînement** (`data/corpus.txt`) dans la langue du groupe :

   ```
   python data/prepare_data.py              # portugais : Dom Casmurro (par défaut)
   python data/prepare_data.py --lang fr    # français : Maria Chapdelaine
   python data/prepare_data.py --lang en    # anglais : Anne of Green Gables
   python data/prepare_data.py --list       # tous les textes disponibles
   ```

   | id | Langue | Livre | Pourquoi ce choix |
   |---|---|---|---|
   | `machado` (par défaut en pt) | portugais | *Dom Casmurro*, Machado de Assis (1899) | Classique brésilien. Utilise l'orthographe de l'édition de 1899 (« elle », « cançado »). |
   | `hemon` (par défaut en fr) | français | *Maria Chapdelaine*, Louis Hémon (1913) | Classique de la littérature québécoise, situé dans le Québec rural. |
   | `verne` | français | *Vingt mille lieues sous les mers*, Jules Verne (1869-1870) | Aventure classique ; un texte long, utile à combiner avec un autre livre. |
   | `montgomery` (par défaut en en) | anglais | *Anne of Green Gables*, L. M. Montgomery (1908) | Classique canadien facile à lire. |
   | `shakespeare` | anglais | *Tiny Shakespeare* (sélection d'Andrej Karpathy) | Le texte de nanoGPT : permet de comparer avec les résultats publiés. |

   Autres options :
   - `--corpus hemon verne` réunit deux textes ou plus dans un seul corpus (utile pour disposer de davantage de texte en français) ;
   - `--file meu_texto.txt` ou `--url https://...` utilise votre propre texte ;
   - `--keep-lines` conserve les sauts de ligne d'origine (par défaut, les romans sont mis en forme avec un paragraphe par ligne).

   Le script supprime l'en-tête et la licence de Project Gutenberg, remplace les espaces typographiques (fréquentes avant `; : ! ?` en français) par des espaces ordinaires et supprime les caractères invisibles. Exécutez-le une fois sur votre ordinateur, puis distribuez le dossier avec le corpus prêt : le groupe n'aura pas besoin d'accès à Internet. Sans `data/corpus.txt`, l'atelier utilise un court texte de démonstration (`data/sample_pt.txt`) et affiche un avertissement.

4. **Lancez l'autotest** (vérifie les 8 laboratoires avec le code de référence, tous les panneaux et un entraînement de 300 étapes) :

   ```
   python tools/selftest.py
   ```

   Il ne modifie ni `bricks/` ni `.workshop/`.

### Kit hors ligne pour la salle

Si le réseau de la salle est peu fiable, apportez les wheels de PyTorch sur une clé USB. Pour chaque combinaison de système et de version de Python utilisée par les étudiants :

```
pip download torch --index-url https://download.pytorch.org/whl/cpu -d wheels/
```

Dans la salle, avec l'environnement virtuel activé :

```
pip install --no-index --find-links wheels torch
```

Le wheel doit correspondre au système (Windows, macOS, Linux), à l'architecture (x86-64 ou ARM) et à la version de Python (3.10, 3.11, 3.12...). **Testez le kit une fois sur une machine vierge** avant l'atelier. Distribuez le dossier du projet avec `data/corpus.txt` déjà préparé.

## Lancer l'atelier

```
python run.py                  # mode Build : utilise le code des étudiants dans bricks/
python run.py --explore        # mode Explore : utilise le code de référence et débloque tous les labs
python run.py --port 8001      # autre port
python run.py --no-browser     # n'ouvre pas automatiquement le navigateur
python run.py --lang pt        # Portuguese guides (default)
python run.py --lang en        # English guides
python run.py --lang fr        # French guides
```

Les 8 guides sont disponibles dans [`workshop/guide/pt/`](workshop/guide/pt/), [`workshop/guide/en/`](workshop/guide/en/) et [`workshop/guide/fr/`](workshop/guide/fr/). `--lang` fonctionne aussi avec `--explore` et change uniquement la langue des guides, sans modifier le code, l'interface en anglais, le corpus ni la progression. Pour changer le texte d'entraînement, exécutez séparément `python data/prepare_data.py --lang pt|en|fr`.

L'adresse est `http://127.0.0.1:8000/`. Le serveur accepte uniquement les connexions de l'ordinateur local, la page ne charge rien depuis Internet et aucune donnée ne quitte la machine. Pour arrêter : **Ctrl+C** dans le terminal.

La progression est enregistrée dans `.workshop/` (`progress.json`, `model.pt`) : au prochain lancement, les laboratoires terminés sont revérifiés silencieusement et le modèle entraîné est rechargé. Pour repartir de zéro, arrêtez le serveur et supprimez le dossier `.workshop/`.

Le **mode Explore** permet au formateur de présenter les panneaux ou à chacun de simplement explorer : il n'utilise ni ne modifie les fichiers de `bricks/` et enregistre son état dans `.workshop/explore/`.

## Parcours de l'étudiant

1. Dans VS Code, ouvrez **le dossier du projet** (Fichier → Ouvrir le dossier). L'enregistrement automatique est déjà activé dans `.vscode/settings.json`.
2. Exécutez `python run.py` et utilisez la page qui s'ouvre dans le navigateur.
3. Dans chaque laboratoire : lisez le guide → cliquez sur **Copy** (Copier) → ouvrez le fichier indiqué (`bricks/bNN_....py`) → collez **sous** la ligne `# ---- PASTE BELOW ----` → revenez à la page et cliquez sur **Verify** (Vérifier).
4. La vérification affiche chaque test avec ✓ ou ✗. Lorsque tous les tests réussissent, la visualisation à droite et le laboratoire suivant sont débloqués.
5. Les étudiants qui souhaitent écrire le code eux-mêmes peuvent cliquer sur **I want to write it** (Je veux écrire le code) pour masquer le code du guide.

## Dépannage

### Messages d'erreur lors de la vérification

Les messages en anglais sont traduits ci-dessous à titre indicatif.

| Message | Cause fréquente | Que faire |
|---|---|---|
| Il n'y a pas encore de code sous la ligne `# ---- PASTE BELOW ----` | Le fichier n'a pas été enregistré ou le code a été collé dans un autre fichier | Appuyez sur Ctrl+S dans VS Code et vérifiez le nom du fichier |
| Guillemets typographiques dans le code | Code copié depuis des diapositives, Word, un PDF ou un chat | Utilisez le bouton **Copy** (Copier) du guide |
| Caractère invisible dans le code (U+00A0, U+200B...) | Code passé par un courriel, un document ou un chat | Supprimez les espaces de la ligne et retapez-les, ou utilisez **Copy** |
| Problème d'indentation / Indentation inattendue | La sélection a perdu (ou gagné) des espaces en début de ligne | Utilisez **Copy** ; dans VS Code, Tab et Maj+Tab ajustent l'indentation |
| La fin du code manque | Sélection incomplète | Utilisez **Copy**, qui copie le bloc entier |
| Ce code semble provenir du Lab N | Code collé dans le mauvais fichier | Coupez-le et collez-le dans le fichier indiqué dans le message |
| PyTorch n'est pas installé dans cet environnement Python | L'environnement virtuel n'a pas été activé ou PyTorch n'a pas été installé | Activez `.venv` et installez PyTorch (voir ci-dessous) |

Les erreurs dans le code (par exemple, `TypeError` ou des formes de tenseurs incompatibles) sont affichées avec le type d'erreur, le message et la ligne du fichier de l'étudiant.

### Terminer cette étape (plan B)

Le lien **Complete this step** (Terminer cette étape), sous le bouton **Verify**, remplace le contenu du fichier du laboratoire par le code de référence, puis lance la vérification. Une copie du fichier précédent est conservée dans `.workshop/backup/`. Utilisez-le lorsqu'une personne est bloquée et que le groupe doit avancer.

### Le port est déjà utilisé

`Port 8000 is already in use` (Le port 8000 est déjà utilisé) : l'atelier est probablement déjà ouvert dans un autre terminal. Fermez-le ou exécutez `python run.py --port 8001`.

### Installation de PyTorch selon le système

- **Windows :** `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Si une erreur de DLL apparaît (par exemple, `WinError 126`), installez le *Microsoft Visual C++ Redistributable* (x64) et redémarrez l'ordinateur.
- **macOS avec Apple Silicon (M1 ou plus récent) :** `pip install torch`.
- **macOS avec processeur Intel :** PyTorch ne publie plus de versions officielles pour Mac Intel après la version 2.2, avec des installateurs uniquement pour Python 3.10 à 3.12. Utilisez Python 3.12 ou antérieur : `pip install torch` installe la version 2.2.2, qui contient tout ce qu'utilise l'atelier. Lancez l'autotest pour confirmer.
- **Linux :** `pip install torch --index-url https://download.pytorch.org/whl/cpu` évite de télécharger la version CUDA, beaucoup plus volumineuse.

Vérifiez toujours que `pip` appartient à l'environnement virtuel : `python -m pip install ...` le garantit.

### La vérification prend du temps

La première vérification du Lab 0 mesure la vitesse de l'ordinateur (environ une demi-seconde). Si la vérification dépasse 20 secondes, le code contient probablement une boucle infinie : arrêtez le serveur (Ctrl+C) et relancez `python run.py`.

## Structure du projet

```
build-your-own-llm/
├── run.py                  # point d'entrée : python run.py [--port 8000] [--explore] [--no-browser]
├── requirements.txt        # uniquement torch
├── bricks/                 # fichiers des étudiants (b00 ... b07), avec la ligne de repère
├── reference/              # code de référence de chaque laboratoire (identique au guide)
├── data/
│   ├── prepare_data.py     # télécharge ou prépare data/corpus.txt
│   └── sample_pt.txt       # court texte de démonstration
├── workshop/               # moteur de l'atelier (serveur, vérifications, entraînement, guides)
│   └── guide/              # guides : pt/, en/, fr/, contenant chacun lab00.md ... lab07.md
├── web/                    # page : index.html, style.css, app.js, charts.js, md.js
├── tools/selftest.py       # autotest pour le formateur
└── .workshop/              # créé à l'exécution : progression, modèle entraîné, sauvegardes
```

Lancez les tests des guides et du choix de langue avec `python -B -m unittest tools.test_guides`. Ils ne nécessitent pas PyTorch ; la vérification du rendu utilise Node.js.

## Prochains laboratoires

| Lab | Thème |
|---|---|
| 8 | Position et masque causal |
| 9 | Self-attention |
| 10 | Multi-head + MLP |
| 11 | Bloc Transformer |
| 12 | Mini-GPT |
| 13 | Playground |

## Crédits

- La structure de l'atelier s'inspire des projets **nanoGPT** et **makemore** d'Andrej Karpathy (licence MIT).
- **Tiny Shakespeare** : texte réuni par Andrej Karpathy dans le projet *char-rnn*, à partir des œuvres de William Shakespeare (domaine public).
- Textes de **Project Gutenberg**, dans le domaine public (auteurs décédés depuis plus de 80 ans) : *Dom Casmurro*, de Machado de Assis (eBook #55752) ; *Maria Chapdelaine*, de Louis Hémon (#13525) ; *Vingt mille lieues sous les mers*, de Jules Verne (#5097) ; *Anne of Green Gables*, de L. M. Montgomery (#45).
- `data/sample_pt.txt` est un texte original écrit pour cet atelier.
