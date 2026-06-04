# Environment Setup

This project uses `uv` to create a reproducible Python environment from
`pyproject.toml` and `uv.lock`.

## Install uv

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create the environment

Install the base environment for data preparation and evaluation:

```bash
uv sync
```

Install the full notebook/modeling environment:

```bash
uv sync --all-extras
```

Install only selected extras:

```bash
uv sync --extra notebook
uv sync --extra model --extra notebook
```

`uv` creates the virtual environment in `.venv/`. This directory is ignored by
Git and should not be committed.

## Run commands inside the environment

```bash
cd app/data
uv run python ori_dataset_analysis.py
uv run python argument_combine.py

cd ../model
uv run python compare_result.py
```

To use notebooks with this environment:

```bash
uv run python -m ipykernel install --user --name veripromiseesg-2026
```

Then choose the `veripromiseesg-2026` kernel in Jupyter, VS Code, or Colab-like
notebook tools.

## API keys

Do not commit real secrets. For LLM data augmentation, create:

```text
app/data/.env
```

with:

```text
API_KEY=your_openai_api_key
```

`app/data/.env` is already ignored by Git.

## GitHub files to commit

Commit these environment files:

```text
pyproject.toml
uv.lock
.python-version
ENVIRONMENT.md
app/data/.env.example
```

Do not commit:

```text
.venv/
app/data/.env
```
