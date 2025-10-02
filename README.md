Word Embeddings for Text Classification: A Performance Evaluation

This project evaluates the performance of various text embedding techniques—GPT-3, GloVe, Word2Vec, and MPNet—for text classification tasks using different machine learning models. The dataset used is the **Fine Food Reviews** dataset.

---

## Table of Contents
1. [Data Importation and Preparation](#1-data-importation-and-preparation)
2. [Embedding Generation](#2-embedding-generation)
    - [GPT-3 Embeddings](#21-gpt-3-embeddings)
    - [GloVe Embeddings](#22-glove-embeddings)
    - [Word2Vec Embeddings](#23-word2vec-embeddings)
    - [MPNet Embeddings](#24-mpnet-embeddings)
3. [Dimensionality Comparison](#3-dimensionality-comparison)
4. [Machine Learning](#4-machine-learning)
5. [Results](#5-results)

---

## 1. Data Importation and Preparation

- The **Fine Food Reviews** dataset is loaded from a GitHub repository. 
- Embedding vectors in the dataset are cleaned using a custom function.
- Columns are renamed for better clarity.

---

## 2. Embedding Generation

Four embedding techniques are used to convert text into numerical vectors:

### 2.1 GPT-3 Embeddings
- Uses OpenAI's `text-embedding-ada-002` model.
- Requires an API key to generate embeddings.

### 2.2 GloVe Embeddings
- GloVe vectors are generated using spaCy's `en_core_web_lg` pipeline.
- A preprocessing function removes redundant full stops before vectorization.

### 2.3 Word2Vec Embeddings
- Pre-trained Word2Vec model (`word2vec-google-news-300`) is loaded using Gensim.
- Tokenization and lemmatization are applied to the text before generating vectors.

### 2.4 MPNet Embeddings
- Uses the `SentenceTransformer` library with the `all-mpnet-base-v2` model for embedding generation.

---

## 3. Dimensionality Comparison

- The dimensionality of embeddings from each technique is compared to highlight differences.

| Embedding   | Dimension |
|-------------|-----------|
| GPT-3       | 1536      |
| MPNet       | 768       |
| Word2Vec    | 300       |
| GloVe       | 300       |

---

## 4. Machine Learning

Four classifiers are used to evaluate the embeddings:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Decision Tree**

- **Pipeline**: Each classifier uses a `RobustScaler` to normalize the data.
- **Metrics**: The primary metric used is classification accuracy.

### Workflow:
1. Split data into training and testing sets.
2. Train each classifier using embeddings as features.
3. Evaluate accuracy and store results.

---


