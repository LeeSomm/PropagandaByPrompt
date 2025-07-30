# Propaganda by Prompt: Repository

This repository contains the code and analysis notebooks accompanying the paper "Propaganda by Prompt: Tracing Hidden Linguistic Strategies in Large Language Models."

## Repository Contents

### Data Requirements

- **ai_propaganda.db**: SQLite database containing the article dataset and LIWC analysis results
  - Must be downloaded from Harvard Dataverse: https://doi.org/10.7910/DVN/KDAWQF
  - Contains both human-written and AI-generated articles
  - Includes LIWC feature analysis for all articles

### Python Scripts

- **lgbm_train_models.py**: Training script for LightGBM propaganda detection models
  - Trains models for different topics and GPT versions
  - Implements feature selection and normalization
  - Produces model performance metrics

### Analysis Notebooks

- **Model_Analysis_Notebook.ipynb**: Primary analysis notebook for model interpretation
  - Loads existing models and queries the database
  - Performs SHAP (SHapley Additive exPlanations) analysis
  - Generates visualizations of feature importance
  - Examines feature interactions and dependencies

### Models Directory

The `models/` directory contains trained LightGBM models (.pkl files) for:
- Combined analysis across all topics
- Topic-specific models (climate, COVID-19, Capitol riot, LGBT)
- Variants with and without punctuation features
- Different GPT versions (GPT-3.5, GPT-4o, GPT-4.1)

## Requirements

Required Python packages are listed in `requirements.txt`. Install using:
```bash
pip install -r requirements.txt
```

## Usage

1. Download the `ai_propaganda.db` file from Harvard Dataverse: https://doi.org/10.7910/DVN/KDAWQF
2. Install required packages
3. Run analysis notebooks for model interpretation
4. Use training scripts to reproduce model results

## Citation

If you use this code or dataset in your research, please cite:
```
[Citation information to be added upon publication]
```

## Contact

For questions about the code or data, please contact the authors through the paper's corresponding author.
