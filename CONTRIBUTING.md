# Contributing to Swahili GPT

Thank you for your interest in contributing! This project is an open research effort to build high-quality language models for African languages. All contributions are welcome — from bug fixes and documentation improvements to training new language variants.

---

## 💡 Ways to Contribute

- **Train a new language variant** — Adapt the notebook to train on Yoruba, Amharic, Hausa, or any other African language.
- **Improve the tokenizer** — Experiment with larger vocabulary sizes or different BPE configurations.
- **Improve `inference.py`** — Add beam search, nucleus (top-p) sampling, or a streaming output mode.
- **Write tests** — Add unit tests for the model architecture or tokenizer pipeline.
- **Fix bugs** — Check the [Issues](https://github.com/casper-justus/swahili-gpt/issues) tab for open bugs.
- **Improve documentation** — Clarify notebook instructions, fix typos, or add examples.

---

## 🛠️ Setting Up Locally

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/swahili-gpt.git
cd swahili-gpt

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a new branch for your changes
git checkout -b feature/your-feature-name
```

---

## 🌍 Training a New Language Variant

The fastest way to adapt this project to a new language:

1. Find a text corpus on [Hugging Face Datasets](https://huggingface.co/datasets) for your target language.
2. Train a new BPE tokenizer on that corpus using the `tokenizers` library.
3. In the notebook, replace the `load_dataset(...)` call and tokenizer path with your new ones.
4. Adjust `VOCAB_SIZE` in Cell 3 to match your new tokenizer.
5. Run the full training pipeline — all other code works as-is.

If you build a variant, please open a PR and we’ll link to it from the README!

---

## 📥 Submitting a Pull Request

1. Make sure your changes work end-to-end before submitting.
2. Keep PRs focused — one feature or fix per PR.
3. Write a clear PR description explaining *what* you changed and *why*.
4. Reference any related issue (e.g. `Closes #12`).
5. Open the PR against the `main` branch.

---

## 💬 Questions?

Open a [GitHub Issue](https://github.com/casper-justus/swahili-gpt/issues) or start a [Discussion](https://github.com/casper-justus/swahili-gpt/discussions). All skill levels are welcome — there are no stupid questions.

---

*Built with ❤️ in Nairobi, Kenya.*
