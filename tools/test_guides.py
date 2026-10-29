"""Guide translation and CLI integration tests; no PyTorch installation required.

Run: python -B -m unittest tools.test_guides
"""
import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import run
from workshop.catalog import DATA_DIR, GUIDE_DIR, GUIDE_LANGUAGES, LABS
from workshop.corpus import CorpusError, load_corpus
from workshop.guide import guide_payload
from workshop.server import create_server
from workshop.session import Session


class GuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(DATA_DIR)

    def test_translation_structure(self):
        self.assertEqual(list(GUIDE_DIR.glob("lab*.md")), [])
        for language in GUIDE_LANGUAGES:
            self.assertEqual(len(list((GUIDE_DIR / language).glob("lab*.md"))), len(LABS))
            for lab in LABS:
                with self.subTest(language=language, lab=lab.number):
                    filename = f"lab{lab.number:02d}.md"
                    original = (GUIDE_DIR / "pt" / filename).read_text(encoding="utf-8")
                    translated = (GUIDE_DIR / language / filename).read_text(encoding="utf-8")
                    self.assertEqual(re.findall(r"^@code\s+(\S+)$", translated, re.M), [lab.brick])
                    self.assertEqual(re.findall(r"^@\w+", translated, re.M),
                                     re.findall(r"^@\w+", original, re.M))
                    self.assertEqual(sorted(re.findall(r"\{\{.*?\}\}", translated)),
                                     sorted(re.findall(r"\{\{.*?\}\}", original)))
                    self.assertEqual(re.findall(r"^#{1,3}(?= )", translated, re.M),
                                     re.findall(r"^#{1,3}(?= )", original, re.M))
                    self.assertEqual(re.findall(r"`[^`]+`", translated),
                                     re.findall(r"`[^`]+`", original))

    def test_payloads_and_rendering(self):
        payloads = []
        for language in GUIDE_LANGUAGES:
            for lab in LABS:
                with self.subTest(language=language, lab=lab.number):
                    warnings = []
                    payload = guide_payload(lab.number, self.corpus, log=warnings.append, language=language)
                    self.assertFalse(payload["fallback"])
                    self.assertEqual(payload["language"], language)
                    self.assertEqual(warnings, [])
                    self.assertNotIn("{{", payload["markdown"])
                    self.assertEqual(list(payload["codes"]), [lab.brick])
                    reference = Path("reference", f"{lab.brick}.py").read_text(encoding="utf-8")
                    self.assertEqual(payload["codes"][lab.brick]["code"], reference)
                    payloads.append(payload)
        script = """
const fs = require('fs');
const path = require('path');
const md = require(path.resolve('web', 'md.js'));
for (const guide of JSON.parse(fs.readFileSync(0, 'utf8'))) {
  const html = md.renderMarkdown(guide.markdown, {codes: guide.codes, lab: guide.lab});
  for (const fragment of ['<h1>', 'class="code-card"', 'class="verify-box"', '<details']) {
    if (!html.includes(fragment)) throw new Error(guide.language + '/' + guide.lab + ': ' + fragment);
  }
  if (/^@(code|verify|answer)/m.test(html)) throw new Error('Unrendered directive');
}
"""
        subprocess.run(["node", "-e", script], input=json.dumps(payloads), text=True, check=True)

    def test_default_and_invalid_languages(self):
        self.assertEqual(guide_payload(0, self.corpus)["language"], "pt")
        for language in ("de", "../en", "", None):
            with self.subTest(language=language):
                with self.assertRaisesRegex(ValueError, "Unsupported guide language"):
                    guide_payload(0, self.corpus, language=language)
                with self.assertRaisesRegex(ValueError, "Unsupported guide language"):
                    Session(language=language)
        with tempfile.TemporaryDirectory() as state_dir:
            self.assertEqual(Session(state_dir=state_dir).language, "pt")

    def test_missing_guide_is_reported(self):
        warnings = []
        with tempfile.TemporaryDirectory() as directory:
            with patch("workshop.guide.GUIDE_DIR", Path(directory)):
                payload = guide_payload(0, self.corpus, log=warnings.append, language="fr")
        self.assertTrue(payload["fallback"])
        self.assertEqual(payload["language"], "fr")
        self.assertIn("workshop/guide/fr/lab00.md", payload["markdown"])
        self.assertIn("@verify", payload["markdown"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("[warning] guide not found:", warnings[0])

    def test_cli_language_selection(self):
        with patch.object(sys, "argv", ["run.py"]):
            self.assertEqual(run.parse_args().lang, "pt")
        for language in GUIDE_LANGUAGES:
            with self.subTest(language=language):
                with patch.object(sys, "argv", ["run.py", "--lang", language, "--explore"]):
                    with patch("workshop.session.Session", side_effect=CorpusError("test stop")) as session:
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(run.main(), 1)
                session.assert_called_once_with(mode="explore", language=language)
        with patch.object(sys, "argv", ["run.py", "--lang", "de"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run.parse_args()
        self.assertEqual(raised.exception.code, 2)

    def test_http_guides_use_session_language(self):
        first_headings = {"pt": "Preparação", "en": "Preparation", "fr": "Préparation"}
        with tempfile.TemporaryDirectory() as directory:
            for language in GUIDE_LANGUAGES:
                with self.subTest(language=language):
                    session = Session(state_dir=Path(directory) / language, language=language)
                    server = create_server(session, 0)
                    worker = threading.Thread(target=server.serve_forever,
                                              kwargs={"poll_interval": 0.05}, daemon=True)
                    worker.start()
                    base = f"http://127.0.0.1:{server.port}"
                    try:
                        with urllib.request.urlopen(base + "/api/state", timeout=5) as response:
                            self.assertEqual(json.load(response)["language"], language)
                        for lab in LABS:
                            with urllib.request.urlopen(base + f"/api/guide/{lab.number}", timeout=5) as response:
                                payload = json.load(response)
                            self.assertEqual(payload["language"], language)
                            self.assertFalse(payload["fallback"])
                            if lab.number == 0:
                                self.assertIn(first_headings[language], payload["markdown"].splitlines()[0])
                    finally:
                        server.shutting_down.set()
                        session.trainer.stop()
                        session.trainer.join(timeout=5)
                        server.shutdown()
                        server.server_close()
                        worker.join(timeout=5)
                    self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
