# ML in Practice I

Welcome 🌙✨ to **ML in Practice I** — a collection of materials, code examples, and exercises for the course.

This repository is public so students and friends can clone it, run the code, and experiment with machine learning in practice.

---

## 📦 Requirements

- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/) (Python package manager, faster than pip)
- Docker (optional, for running MongoDB or other services)

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/ajakovac/ML-in-Practice-I.git
cd ML-in-Practice-I
```

### 2. Install `uv`

On Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sudo sh
```

On Windows:

```
irm https://astral.sh/uv/install.ps1 | iex
```

sync dependencies
```
uv sync
```

This will create an environment, too.

### 3. Activate the environment

Linux/macOS:
```bash
source .venv/bin/activate
```
On windows:
```
.venv\Scripts\Activate.ps1
```


## 🐋 Running MongoDB (optional)

To use MongoDB for database examples:

- The `invoke mongo` task will:
  - Create a `Data/mongo/` folder if needed
  - Pull and run the `mongo:7.0` Docker image
  - Mount the local directory for persistence

- You can stop MongoDB with the `invoke stop-mongo` task.
