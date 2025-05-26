# Mitigating Bias in AI using Large Language Models (LLMs)

This repository provides tools and examples for analyzing and mitigating **demographic bias** in NLP models using two key datasets: **BUG** (Bias in Utterance Generation) and **EEC** (Equity Evaluation Corpus). The aim is to help researchers and practitioners understand, visualize, and address gender and racial bias in language models.

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
  - [BUG Dataset](#bug-dataset)
  - [EEC Dataset](#eec-dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Example Analyses](#example-analyses)
- [Dependencies](#dependencies)
- [License](#license)
- [References](#references)

---

## Overview

Modern NLP systems can unintentionally reflect or amplify societal biases. This project demonstrates how to use the BUG and EEC datasets to analyze such biases, especially those related to gender and race, and lays the groundwork for developing debiasing strategies in AI.

---

## Datasets

### BUG Dataset

The **BUG (Bias in Utterance Generation)** dataset focuses on **gender bias** in sentences where professions are associated with gendered pronouns. It is ideal for evaluating how models handle stereotypical, neutral, and anti-stereotypical profession-pronoun pairings.

**Key Columns:**

| Column Name              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `sentence_text`          | Full sentence containing at least one profession and gendered pronoun.      |
| `tokens`                 | Tokenized version of the sentence.                                          |
| `profession`             | Profession found in the sentence (e.g., doctor, nurse).                     |
| `g`                      | Gendered pronoun (e.g., "his", "her").                                      |
| `profession_first_index` | Token index of the profession in the sentence.                              |
| `g_first_index`          | Token index of the gendered pronoun.                                        |
| `predicted gender`       | Inferred gender from the pronoun ("Male" or "Female").                      |
| `stereotype`             | Bias label: -1 (anti-stereotypical), 0 (neutral), 1 (stereotypical).        |
| `distance`               | Number of tokens between profession and pronoun.                            |
| `num_of_pronouns`        | Count of gendered pronouns in the sentence.                                 |
| `corpus`                 | Original source corpus (e.g., "covid19", "bios").                           |
| `data_index`             | Unique identifier for the data entry.                                       |

**Interpretation Example:**

- `sentence_text`: "Her early years as a resident doctor..."
- `profession`: doctor
- `g`: her
- `predicted gender`: Female
- `stereotype`: -1

This is **anti-stereotypical** because "doctor" is stereotypically male, but the pronoun is female.

**Use Cases:**
- Analyzing gender bias in NLP models
- Training and evaluating debiasing algorithms
- Enhancing fairness in applications (hiring, education, healthcare, etc.)

---

### EEC Dataset

The **Equity Evaluation Corpus (EEC)** is a benchmark dataset for evaluating **demographic bias** (gender and race) in NLP, especially in the context of emotion and sentiment analysis.

**Key Columns:**

| Column Name    | Description                                                     |
|----------------|-----------------------------------------------------------------|
| `sentence`     | Full sentence (e.g., "Alonzo feels angry.")                     |
| `template`     | Sentence template (e.g., " feels .")   |
| `person`       | Name or subject in the sentence (e.g., "Alonzo")                |
| `gender`       | Gender of the subject ("male", "female")                        |
| `race`         | Race or ethnicity of the subject ("African-American", etc.)     |
| `emotion`      | Emotion category ("anger", "joy", "fear", "sadness")            |
| `emotion word` | Specific emotion word used (e.g., "angry", "furious")           |

**Interpretation Example:**

| sentence              | person  | gender | race             | emotion | emotion word |
|-----------------------|---------|--------|------------------|---------|--------------|
| Alonzo feels angry.   | Alonzo  | male   | African-American | anger   | angry        |
| Amanda feels joyful.  | Amanda  | female | European         | joy     | joyful       |

**Use Cases:**
- Detecting if models assign different scores to sentences based on gender/race
- Auditing fairness in emotion or sentiment analysis
- Benchmarking and improving model equity

---

## Installation

```bash
pip install pandas matplotlib seaborn scikit-learn datasets
```

---

## Usage

### 1. Clone and Prepare the BUG Dataset

```bash
git clone https://github.com/SLAB-NLP/BUG.git
cd BUG
tar -xvzf data.tar.gz
```

### 2. Load the BUG Dataset

```python
import pandas as pd

bug_df = pd.read_csv('data/gold_BUG.csv')
bug_df.drop(columns=['Unnamed: 0', 'uid'], inplace=True)
```

### 3. Load the EEC Dataset

The EEC dataset is available from [Huggig Face Datasets](https://huggingface.co/datasets/peixian/equity_evaluation_corpus):

```python
from datasets import load_dataset

eec = load_dataset("peixian/equity_evaluation_corpus", split="test")
eec_df = eec.to_pandas()
```

---

## Example Analyses

- **BUG**: Analyze the distribution of stereotype labels, visualize profession-gender associations, and evaluate model predictions for bias.
- **EEC**: Compare model emotion/sentiment scores for sentences differing only by gender or race, and visualize disparities.

---

## Dependencies

- pandas
- matplotlib
- seaborn
- scikit-learn
- datasets (Hugging Face & GitHub)

---

## License

- **BUG**: See [BUG GitHub repository](https://github.com/SLAB-NLP/BUG) for license.
- **EEC**: See [EEC GitHub repository](https://huggingface.co/datasets/peixian/equity_evaluation_corpus) for license.

---

## References

- [BUG: Bias in Utterance Generation](https://github.com/SLAB-NLP/BUG)
- [EEC: Equity Evaluation Corpus](https://huggingface.co/datasets/peixian/equity_evaluation_corpus)

