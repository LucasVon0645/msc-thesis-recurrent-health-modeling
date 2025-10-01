# Recurrent Health Events Prediction

This project uses machine learning techniques to predict recurrent health events, such as hospital readmissions, based on structured clinical data.

## 📦 Project Setup

### 1. Clone the Repository

```bash
git clone git@gitlab.com:LucasVonAncken/master-thesis-recurrent-health-events-prediction.git
cd master-thesis-recurrent-health-events-prediction
```

### 2. Set Up the Virtual Environment with Poetry

Make sure you have [Poetry installed](https://python-poetry.org/docs/#installation).

To create a virtual environment **inside the project directory** (`.venv/`):

```bash
poetry config virtualenvs.in-project true --local
```

Then install dependencies:

```bash
poetry install
```

### 3. Activate the Environment

```bash
poetry shell
```

---

## 📦 Optional Setup: Using a Dev Container (Recommended for Reproducibility)

If you’re using Visual Studio Code and want a fully reproducible, OS-independent development environment, you can use the preconfigured Dev Container.

This ensures that all tools, dependencies, and Python versions are installed identically across machines — ideal for research and collaboration. 

⚠️ **Note**: The first build of the container may take a long time (e.g., ~20–25 minutes).This is normal — the container compiles Python from source and installs scientific libraries with native code. Subsequent launches will be much faster thanks to Docker caching.

### 🔧 To Use the Dev Container:

1. Install Docker Desktop

2. Install Visual Studio Code

3. Install the Dev Containers extension in VS Code

4. Open the project folder in VS Code

5. Press Cmd+Shift+P (Mac) or Ctrl+Shift+P (Windows/Linux)

6. Select: *Dev Containers: Reopen in Container*

VS Code will build the environment and reopen your project inside a fully configured container.

## 🚀 Running the Project

Once the environment is activated, you can run any of the project's main scripts. Check the `src/` folder.

---

## 💾 Downloading the MIMIC Dataset

To download the necessary [MIMIC](https://physionet.org/content/mimiciii/) files, run the following script:

```bash
bash scripts/download_mimic_files.sh
```

> 🔐 **Note:** You must have valid PhysioNet credentials and appropriate permissions to access the MIMIC dataset. Refer to the [official MIMIC documentation](https://physionet.org/content/mimiciii/view-only/) for instructions on how to gain access.

---

## 📁 Project Structure

```text
.
├── scripts/                # Utility scripts (e.g., for data download)
├── src/                    # Main source code for training and prediction
├── data/                   # Placeholders for datasets (not tracked)
├── pyproject.toml          # Poetry dependency configuration
├── poetry.lock             # Locked dependency versions for reproducibility
└── README.md
```

---

## ✅ Requirements

- Python 3.9+
- Poetry
- Access to the MIMIC-III or MIMIC-IV dataset

---

## 📚 References

- [Poetry Documentation](https://python-poetry.org/docs/)
- [MIMIC Dataset Info](https://physionet.org/mimic/)