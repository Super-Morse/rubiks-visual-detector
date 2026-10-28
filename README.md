# 🧩 Rubik's Visual Detector

<p align="center">
  <strong>Computer Vision microservice for automated Rubik's Cube state recognition from images.</strong>
</p>

<p align="center">
  <!-- CI/CD & SonarCloud Badges -->
  <a href="https://github.com/Super-Morse/rubiks-visual-detector/actions/workflows/sonarcloud.yml">
    <img src="https://github.com/Super-Morse/rubiks-visual-detector/actions/workflows/sonarcloud.yml/badge.svg" alt="CI Build" />
  </a>
  <a href="https://sonarcloud.io/summary/new_code?id=Super-Morse_rubiks-visual-detector">
    <img src="https://sonarcloud.io/api/project_badges/measure?project=Super-Morse_rubiks-visual-detector&metric=alert_status" alt="Quality Gate Status" />
  </a>
  <a href="https://sonarcloud.io/summary/new_code?id=Super-Morse_rubiks-visual-detector">
    <img src="https://sonarcloud.io/api/project_badges/measure?project=Super-Morse_rubiks-visual-detector&metric=coverage" alt="Coverage" />
  </a>
  <a href="https://sonarcloud.io/summary/new_code?id=Super-Morse_rubiks-visual-detector">
    <img src="https://sonarcloud.io/api/project_badges/measure?project=Super-Morse_rubiks-visual-detector&metric=vulnerabilities" alt="Vulnerabilities" />
  </a>
  <a href="https://sonarcloud.io/summary/new_code?id=Super-Morse_rubiks-visual-detector">
    <img src="https://sonarcloud.io/api/project_badges/measure?project=Super-Morse_rubiks-visual-detector&metric=code_smells" alt="Code Smells" />
  </a>
  <br />
  <!-- Tech Stack Badges -->
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/OpenCV-Headless-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Code%20Style-Ruff-black?style=flat" alt="Ruff" />
</p>

---

## 📖 Overview

**Rubik's Visual Detector** is a high-performance Python microservice designed to process real-world photos of a Rubik's Cube, detect the 3x3 sticker grid, accurately classify colors across varying lighting conditions, and validate the cube's physical state.

This service acts as the computer vision ingestion pipeline for the [Super-Morse/rubiks-solver](https://github.com/Super-Morse/rubiks-solver) — an interactive 3D solver powered by **Go**, **WebAssembly**, and **Three.js**.

---

## 🏗️ Architecture & Engineering Standards

The codebase strictly adheres to **Clean Architecture** and **SOLID** principles:

```text
rubiks-visual-detector/
├── app/
│   ├── api/                 # Presentation layer (FastAPI routes & schemas)
│   ├── core/                # Application configurations & settings
│   ├── domain/              # Business entities, Enums, and Abstract Interfaces
│   ├── infrastructure/      # OpenCV image processing, contour detection & color classifiers
│   └── services/            # Use cases orchestrating domain & infrastructure logic
└── tests/                   # Deterministic test suites (Classic TDD)
```

### Key Engineering Highlights:
* **SOLID Compliance**: Low coupling achieved via Dependency Inversion (`IColorClassifier`, `IGridDetector` abstractions).
* **Test-Driven Development (TDD)**: Developed using the classical (Detroit) school of TDD without fragile mocks. Uses synthetic NumPy pixel buffers and deterministic image fixtures.
* **Perceptual Color Science**: Classifies sticker colors using **CIELAB Delta-E (CIE2000)** and clustering instead of brittle RGB/HSV hardcoded ranges, making it robust against shadows and ambient light variations.
* **Continuous Quality**: Integrated with **SonarCloud** and **GitHub Actions** checking coverage, cognitive complexity, and security vulnerabilities on every push.

---

## 🚀 Quick Start

### Option 1: Running with Docker (Recommended)

The project includes an optimized multi-layer Docker setup using a non-root user:

```bash
# Clone the repository
git clone https://github.com/Super-Morse/rubiks-visual-detector.git
cd rubiks-visual-detector

# Spin up the service
docker compose up --build
```

The API will be available at:
* **Interactive Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Alternative Docs (ReDoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

### Option 2: Local Development

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Start the dev server
uvicorn app.main:app --reload --port 8000
```

---

## 🧪 Testing & Code Quality

Run tests, coverage reports, and linters with ease:

```bash
# Run unit & integration tests with coverage report
pytest

# Format code according to PEP 8 using Ruff
ruff format .

# Fast linting and auto-fix
ruff check --fix .

# Static type analysis
mypy app/
```

---

## 🔗 Related Projects

* **[rubiks-solver](https://github.com/Super-Morse/rubiks-solver)** — The client-side Rubik's Cube solver utilizing Go (Kociemba two-phase algorithm) compiled to WebAssembly with Three.js rendering.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
