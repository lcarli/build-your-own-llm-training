"""Static description of the labs: brick files, titles and the names each brick defines."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRICKS_DIR = ROOT / "bricks"
REFERENCE_DIR = ROOT / "reference"
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"
GUIDE_DIR = ROOT / "workshop" / "guide"
GUIDE_LANGUAGES = ("pt", "en", "fr")
DEFAULT_GUIDE_LANGUAGE = "pt"
STATE_DIR = ROOT / ".workshop"

MARKER = "# ---- PASTE BELOW ----"

# Fixed hyper-parameters of the bigram labs (also used as guide template variables).
N_EMBD = 32
BLOCK_SIZE = 8              # context length used by the training loop
MODEL_SEED = 1337           # seed of the session model created at Lab 4
SAMPLE_SEED = 42            # seed of the "evolution gallery" samples
SAMPLE_CHARS = 200          # length of every generated sample
EVAL_INTERVAL = 250         # evaluate train/val loss every N training steps
EVAL_BATCHES = 40           # batches averaged per evaluation
SAMPLE_STEPS = (500, 1000, 2000)   # extra gallery checkpoints (plus step 0 and the final step)
TRAIN_FRACTION = 0.9
TORCH_INSTALL_CMD = "pip install torch --index-url https://download.pytorch.org/whl/cpu"


@dataclass(frozen=True)
class LabInfo:
    number: int
    brick: str          # file stem inside bricks/ and reference/
    title: str          # title shown in the UI
    short: str          # short label for the navigation chips
    english: str        # English title used in the brick file header
    names: tuple        # top-level names the brick must define
    kinds: tuple        # "function" or "class" for each name (used in messages)

    @property
    def module_name(self):
        return f"bricks.{self.brick}"

    @property
    def rel_path(self):
        return f"bricks/{self.brick}.py"


LABS = (
    LabInfo(0, "b00_warmup", "Preparation: the workshop workflow", "Preparation", "Warm-up",
            ("greet",), ("function",)),
    LabInfo(1, "b01_tokenizer", "Tokenizer: text becomes numbers", "Tokenizer", "Character tokenizer",
            ("CharTokenizer",), ("class",)),
    LabInfo(2, "b02_batches", "Training examples", "Batches", "Training batches",
            ("split_train_val", "get_batch"), ("function", "function")),
    LabInfo(3, "b03_embeddings", "Embeddings: each token becomes a vector", "Embeddings", "Token embeddings",
            ("TokenEmbedding",), ("class",)),
    LabInfo(4, "b04_bigram", "Logits: the first prediction", "Logits", "Bigram model (logits)",
            ("BigramLM",), ("class",)),
    LabInfo(5, "b05_sampling", "Softmax, temperature and generation", "Generation", "Softmax and sampling",
            ("next_token_probs", "generate"), ("function", "function")),
    LabInfo(6, "b06_loss", "Loss: measuring the error", "Loss", "Loss (cross-entropy)",
            ("next_token_loss",), ("function",)),
    LabInfo(7, "b07_training", "Training: the model learns", "Training", "Training loop",
            ("make_optimizer", "train_step"), ("function", "function")),
)

FUTURE_LABS = (
    (8, "Position and causal mask", "Position"),
    (9, "Self-attention", "Attention"),
    (10, "Multi-head + MLP", "Multi-head"),
    (11, "Transformer block", "Block"),
    (12, "Mini-GPT", "Mini-GPT"),
    (13, "Playground", "Playground"),
)

LAST_LAB = LABS[-1].number

# Name -> lab number, used to detect code pasted into the wrong file.
NAME_TO_LAB = {name: lab.number for lab in LABS for name in lab.names}


def lab_info(n):
    """Return the LabInfo for lab n (raises KeyError for unknown labs)."""
    if not isinstance(n, int) or not 0 <= n <= LAST_LAB:
        raise KeyError(n)
    return LABS[n]


def brick_header(lab):
    """The pre-created header of a brick file (everything up to and including the marker)."""
    return (
        f"# Build Your Own LLM - Lab {lab.number}: {lab.english}\n"
        "# Paste the code from the lab guide below the marker line.\n"
        "# The workshop reads this file when you click \"Verify\".\n"
        f"{MARKER}\n"
    )
