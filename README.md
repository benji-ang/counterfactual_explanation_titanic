# Explainability in AI — Titanic

This project implements and evaluates multiple explainability techniques on a neural network trained to predict Titanic passenger survival. The Titanic dataset serves as a lens for examining social biases learned by ML models.

## Overview

A 3-layer ReLU network (7 → 256 → 256 → 1) is trained on the Titanic dataset and then analysed with:

- **Feature attributions** — custom SHAP implementation plus Captum's Shapley Value Sampling and DeepLIFT, with quantitative infidelity evaluation and a computational efficiency comparison on the [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Counterfactual explanations** — Nearest-Neighbour Counterfactual Explanations (NNCE) and the gradient-based Wachter et al. (2017) method (WAC), evaluated on validity, proximity, and plausibility

## Repository structure

```
titanic_explainability.ipynb   # main notebook (all tasks)
titanic-dataset.csv            # raw Titanic passenger data
requirements.txt               # Python dependencies
report.pdf                     # written analysis
```

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then launch the notebook:

```bash
jupyter lab titanic_explainability.ipynb
```

Run cells top-to-bottom — data loading, EDA, model training, and explanation sections are all in sequence.

## Notebook sections

| Section | Content |
|---|---|
| Data Loading & Preprocessing | `InvertibleColumnTransformer`, `TitanicDataset` |
| Exploratory Data Analysis | Pearson correlation heatmap, survival by sex |
| Model Training | Train and evaluate the neural network |
| Feature Attributions | SHAP (from scratch), SVS & DeepLIFT (Captum), infidelity, efficiency on Dry Bean |
| Counterfactual Explanations | Distance metric design, NNCE, WAC, metric comparison |

## Key dependencies

| Package | Purpose |
|---|---|
| `torch` | Model training and gradient-based counterfactuals |
| `captum` | Shapley Value Sampling, DeepLIFT, infidelity metric |
| `scikit-learn` | Preprocessing pipeline |
| `pandas` / `numpy` | Data handling |
| `matplotlib` / `seaborn` | Visualisations |
