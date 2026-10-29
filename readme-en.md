# Build Your Own LLM — Interactive Lab

[Português](readme-pt.md) | [English](readme-en.md) | [Français](readme-fr.md)

**Project convention:** always write code comments, docstrings, and messages in English. The interface uses English; guides and training texts retain their own languages.

A guided workshop where each student assembles a **character-level language model**, piece by piece (a bigram in this version; mini-GPT will come in later labs). Everything runs **locally, on the CPU**: no cloud, no GPU, no paid services, and no GitHub Copilot.

In each lab, a local page displays a guide with explanations and ready-to-use code. The student **copies** it, **pastes** it into the indicated file in VS Code, and clicks **Verify**. The page then unlocks a visualization showing **real** numbers from their model: no made-up data.

| Lab | Topic | Student file |
|---|---|---|
| 0 | Preparation: the workshop workflow | `bricks/b00_warmup.py` |
| 1 | Tokenizer: turning text into numbers | `bricks/b01_tokenizer.py` |
| 2 | Training examples (batches) | `bricks/b02_batches.py` |
| 3 | Embeddings | `bricks/b03_embeddings.py` |
| 4 | Logits: the first prediction | `bricks/b04_bigram.py` |
| 5 | Softmax, temperature, and generation | `bricks/b05_sampling.py` |
| 6 | Loss: measuring the error | `bricks/b06_loss.py` |
| 7 | Training | `bricks/b07_training.py` |

## Requirements

- **Python 3.10 or newer** (Windows, macOS, or Linux).
- **PyTorch** (CPU version). This is the only dependency: the server uses only the Python standard library, and the page is plain HTML/CSS/JS with no external libraries.
- A modern browser (Chrome, Edge, Firefox, or Safari) and **VS Code**.
- Any ordinary laptop: the model has only a few thousand parameters and trains in seconds.

## Setup (instructor)

1. **Create a virtual environment** in the project folder:

   | System | Commands |
   |---|---|
   | Windows (PowerShell) | `py -3 -m venv .venv` and `.venv\Scripts\Activate.ps1` |
   | Windows (cmd) | `py -3 -m venv .venv` and `.venv\Scripts\activate.bat` |
   | macOS / Linux | `python3 -m venv .venv` and `source .venv/bin/activate` |

   If PowerShell blocks the activation script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

2. **Install PyTorch for CPU** (a much smaller download than the GPU version):

   ```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Prepare the training text** (`data/corpus.txt`) in the class's language:

   ```
   python data/prepare_data.py              # Portuguese: Dom Casmurro (default)
   python data/prepare_data.py --lang fr    # French: Maria Chapdelaine
   python data/prepare_data.py --lang en    # English: Anne of Green Gables
   python data/prepare_data.py --list       # all available texts
   ```

   | id | Language | Book | Why this one |
   |---|---|---|---|
   | `machado` (pt default) | Portuguese | *Dom Casmurro*, Machado de Assis (1899) | Brazilian classic. Uses the spelling of the 1899 edition ("elle", "cançado"). |
   | `hemon` (fr default) | French | *Maria Chapdelaine*, Louis Hémon (1913) | A classic of Quebec literature, set in rural Quebec. |
   | `verne` | French | *Vingt mille lieues sous les mers*, Jules Verne (1869-1870) | Classic adventure; a long text, useful for combining with another book. |
   | `montgomery` (en default) | English | *Anne of Green Gables*, L. M. Montgomery (1908) | An easy-to-read Canadian classic. |
   | `shakespeare` | English | *Tiny Shakespeare* (selection by Andrej Karpathy) | The nanoGPT text: allows comparisons with published results. |

   Other options:
   - `--corpus hemon verne` combines two or more texts into a single corpus (useful for getting more French text);
   - `--file meu_texto.txt` or `--url https://...` uses your own text;
   - `--keep-lines` preserves the original line breaks (by default, novels are formatted with one paragraph per line).

   The script removes the Project Gutenberg header and license, replaces typographic spaces (common before `; : ! ?` in French) with regular spaces, and removes invisible characters. Run it once on your machine, then distribute the folder with the corpus ready: the class will not need internet access. Without `data/corpus.txt`, the workshop uses a short demonstration text (`data/sample_pt.txt`) and displays a warning.

4. **Run the self-test** (checks all 8 labs using the reference code, all panels, and a 300-step training run):

   ```
   python tools/selftest.py
   ```

   It does not modify `bricks/` or `.workshop/`.

### Offline classroom package

If the classroom network is unreliable, bring PyTorch wheels on a USB drive. For each combination of operating system and Python version used by students:

```
pip download torch --index-url https://download.pytorch.org/whl/cpu -d wheels/
```

In the classroom, with the virtual environment activated:

```
pip install --no-index --find-links wheels torch
```

The wheel must match the operating system (Windows, macOS, Linux), architecture (x86-64 or ARM), and Python version (3.10, 3.11, 3.12...). **Test the package on a clean machine once** before the workshop. Distribute the project folder with `data/corpus.txt` already prepared.

## Running the workshop

```
python run.py                  # Build mode: uses students' code in bricks/
python run.py --explore        # Explore mode: uses reference code and unlocks all labs
python run.py --port 8001      # another port
python run.py --no-browser     # does not open the browser automatically
python run.py --lang pt        # Portuguese guides (default)
python run.py --lang en        # English guides
python run.py --lang fr        # French guides
```

All 8 guides are available in [`workshop/guide/pt/`](workshop/guide/pt/), [`workshop/guide/en/`](workshop/guide/en/), and [`workshop/guide/fr/`](workshop/guide/fr/). `--lang` also works with `--explore` and changes only the guide language, not the code, English interface, corpus, or progress. To change the training text, separately run `python data/prepare_data.py --lang pt|en|fr`.

The address is `http://127.0.0.1:8000/`. The server only accepts connections from the local computer, the page loads nothing from the internet, and no data leaves the machine. To stop: **Ctrl+C** in the terminal.

Progress is stored in `.workshop/` (`progress.json`, `model.pt`): on the next run, completed labs are silently checked again and the trained model is reloaded. To start over, stop the server and delete the `.workshop/` folder.

**Explore mode** lets the instructor demonstrate the panels or anyone simply explore: it neither uses nor changes files in `bricks/` and stores its state in `.workshop/explore/`.

## Student workflow

1. In VS Code, open **the project folder** (File → Open Folder). Auto Save is already enabled in `.vscode/settings.json`.
2. Run `python run.py` and use the page that opens in the browser.
3. In each lab: read the guide → click **Copy** → open the indicated file (`bricks/bNN_....py`) → paste **below** the `# ---- PASTE BELOW ----` line → return to the page and click **Verify**.
4. Verification displays each test with ✓ or ✗. Once all tests pass, the visualization on the right and the next lab are unlocked.
5. Students who want to write the code themselves can click **I want to write it** to hide the guide's code.

## Troubleshooting

### Error messages during verification

The messages below summarize common verification errors.

| Message | Common cause | What to do |
|---|---|---|
| No code below the `# ---- PASTE BELOW ----` line yet | The file was not saved, or the code was pasted into another file | Press Ctrl+S in VS Code and check the file name |
| Curly quotes in the code | Code copied from slides, Word, PDF, or chat | Use the guide's **Copy** button |
| Invisible character in the code (U+00A0, U+200B...) | Code passed through email, a document, or chat | Delete the spaces on the line and type them again, or use **Copy** |
| Indentation problem / Unexpected indentation | The selection lost (or gained) spaces at the beginning of lines | Use **Copy**; in VS Code, Tab and Shift+Tab adjust indentation |
| The end of the code is missing | Incomplete selection | Use **Copy**, which copies the entire block |
| This code appears to be from Lab N | Code pasted into the wrong file | Cut and paste it into the file indicated in the message |
| PyTorch is not installed in this Python environment | The virtual environment was not activated, or PyTorch was not installed | Activate `.venv` and install PyTorch (see below) |

Errors within the code (for example, `TypeError` or incompatible tensor shapes) are displayed with the error type, message, and line in the student's file.

### Complete this step (plan B)

The **Complete this step** link below the Verify button replaces the lab file's contents with the reference code and then verifies it. A copy of the previous file is saved in `.workshop/backup/`. Use this when someone gets stuck and the group needs to move on.

### The port is already in use

`Port 8000 is already in use`: the workshop is probably open in another terminal. Close it or run `python run.py --port 8001`.

### Installing PyTorch by operating system

- **Windows:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`. If a DLL error appears (for example, `WinError 126`), install the *Microsoft Visual C++ Redistributable* (x64) and restart the computer.
- **macOS with Apple Silicon (M1 or newer):** `pip install torch`.
- **macOS with an Intel processor:** official PyTorch stopped publishing Intel Mac versions after 2.2, with installers only for Python 3.10 through 3.12. Use Python 3.12 or earlier: `pip install torch` installs 2.2.2, which includes everything the workshop uses. Run the self-test to confirm.
- **Linux:** `pip install torch --index-url https://download.pytorch.org/whl/cpu` avoids downloading the much larger CUDA version.

Always check that `pip` belongs to the virtual environment: `python -m pip install ...` ensures this.

### Verification takes too long

The first Lab 0 verification measures the computer's speed (about half a second). If verification takes longer than 20 seconds, the code probably has an infinite loop: stop the server (Ctrl+C) and run `python run.py` again.

## Project structure

```
build-your-own-llm/
├── run.py                  # entry point: python run.py [--port 8000] [--explore] [--no-browser]
├── requirements.txt        # torch only
├── bricks/                 # student files (b00 ... b07), with the marker line
├── reference/              # reference code for each lab (same as the guide)
├── data/
│   ├── prepare_data.py     # downloads or prepares data/corpus.txt
│   └── sample_pt.txt       # short demonstration text
├── workshop/               # workshop engine (server, checks, training, guides)
│   └── guide/              # guides: pt/, en/, fr/, each containing lab00.md ... lab07.md
├── web/                    # page: index.html, style.css, app.js, charts.js, md.js
├── tools/selftest.py       # instructor self-test
└── .workshop/              # created at runtime: progress, trained model, backups
```

Run guide and language-selection tests with `python -B -m unittest tools.test_guides`. They do not require PyTorch; the renderer check uses Node.js.

## Upcoming labs

| Lab | Topic |
|---|---|
| 8 | Position and causal masking |
| 9 | Self-attention |
| 10 | Multi-head + MLP |
| 11 | Transformer block |
| 12 | Mini-GPT |
| 13 | Playground |

## Credits

- The workshop structure is inspired by Andrej Karpathy's **nanoGPT** and **makemore** projects (MIT license).
- **Tiny Shakespeare**: text compiled by Andrej Karpathy in the *char-rnn* project, from William Shakespeare's works (public domain).
- **Project Gutenberg** public-domain texts (authors who died more than 80 years ago): *Dom Casmurro*, by Machado de Assis (eBook #55752); *Maria Chapdelaine*, by Louis Hémon (#13525); *Vingt mille lieues sous les mers*, by Jules Verne (#5097); *Anne of Green Gables*, by L. M. Montgomery (#45).
- `data/sample_pt.txt` is an original text written for this workshop.
