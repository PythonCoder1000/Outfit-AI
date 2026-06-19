# Outfit AI

An AI-powered outfit assistant — generate, recommend, and reason about clothing
outfits using modern language and vision models.

## Features

- AI-driven outfit suggestions and styling recommendations
- Configurable model backends via environment variables
- Simple, scriptable Python interface

> This project is in early development. The feature set above is the intended
> direction; see the source for what is currently implemented.

## Requirements

- Python 3.10+
- An API key for your chosen model provider (stored in `.env`)

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/PythonCoder1000/Outfit-AI.git
cd Outfit-AI

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt  # if/when a requirements file is present

# 4. Configure secrets
cp .env.example .env
# then edit .env and fill in your keys
```

## Configuration

Secrets and runtime configuration are read from a local `.env` file, which is
**git-ignored** and must never be committed. Copy `.env.example` to `.env` and
fill in your own values:

```env
# .env
ANTHROPIC_API_KEY=your-key-here
```

## Usage

```bash
python main.py
```

## Project Structure

```
Outfit-AI/
├── .env.example      # Template for required environment variables
├── .gitignore        # Ignores .env, caches, venvs, build artifacts
└── README.md
```

## License

This project is currently unlicensed. Add a `LICENSE` file to define usage terms.
