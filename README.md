# LinguaLens Backend

A FastAPI-based backend service, exposed externally via ngrok.

---

## Prerequisites

- Python 3.8+
- [ngrok](https://ngrok.com) installed and authenticated

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
source .venv/Scripts/activate
```

### 3. Add `.gitignore`

```bash
echo ".venv" > .gitignore
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Server

### Start the API

From the `lingualens_backend/src` folder:

```bash
cd src
uvicorn main:app --reload --port 8000
```

### Expose with ngrok

In a separate terminal, from the `lingualens_backend` folder:

```bash
ngrok http 8000
```

ngrok will give you a public URL like:
```
https://abc123.ngrok-free.app -> http://localhost:8000
```

Verify it's working:

```bash
curl https://abc123.ngrok-free.app/health
# Expected response: {"status": "ok"}
```

---

## Deactivating the Virtual Environment

```bash
deactivate
```

---

## Troubleshooting

### `AttributeError: Can't get attribute 'Vocabulary'`

**Full error:**
```
AttributeError: Can't get attribute 'Vocabulary' on <module '__mp_main__' from '.../lingualens_backend/.venv/bin/uvicorn'>
```

**Cause:** pickle cannot locate the `Vocabulary` class when loading a saved model under uvicorn.

**Fix:** Add the following to `main.py` to register the class in the correct module namespace:

```python
# Force pickle to find Vocabulary in the right place
sys.modules['__main__'].Vocabulary = Vocabulary
```

---

## References

- [FastAPI Virtual Environments](https://fastapi.tiangolo.com/virtual-environments/)
- [ngrok Dashboard](https://dashboard.ngrok.com/)