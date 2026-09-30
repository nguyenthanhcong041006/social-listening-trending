# Social Listening & Trend Prediction Pipeline

An end-to-end multilingual **Social Listening** and **Topic Trend Prediction** system leveraging state-of-the-art deep learning architectures including **XLM-RoBERTa**, **BERTopic**, time-series forecasting with **SARIMA**, and gradient boosting classification via **LightGBM**.

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Pipeline Architecture Diagram](#2-pipeline-architecture-diagram)
3. [Repository Directory Structure & File Manifest](#3-repository-directory-structure--file-manifest)
4. [Detailed Breakdown of Pipeline Stages & Steps](#4-detailed-breakdown-of-pipeline-stages--steps)
5. [Core Mathematical Formulations & Algorithms](#5-core-mathematical-formulations--algorithms)
6. [.gitignore Policy (Media & Heavy Artifact Protection)](#6-gitignore-policy-media--heavy-artifact-protection)
7. [Installation & Execution Guide](#7-installation--execution-guide)
8. [Notebooks & Visual Analytics Guide](#8-notebooks--visual-analytics-guide)

---

## 1. Project Overview

This repository provides a modular, production-grade machine learning pipeline designed to monitor social media discourse across multiple domains and forecast emerging topics before they reach peak popularity:

- **Live Multi-Domain Social Ingestion**: Collects 100% genuine posts from Reddit discussions and YouTube channels across 5 domains (Technology, Sports, Gaming, AI, Entertainment) spanning a strict 30-day temporal window with rich engagement metadata (text, timestamps, likes, shares, comments, hashtags, followers).
- **Multi-Step Preprocessing Chain**: Deduplicates posts, filters spam bots, cleans markup/URLs, standardizes Unicode (NFKC) and casing, while preserving essential hashtags, sentiment emojis, and multilingual negation context (`not`, `no`, `never`, `không`, `chẳng`, `chưa`...).
- **Multilingual Representation & Sentiment Polarity**: Employs **XLM-RoBERTa** (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) to concurrently generate pooled contextual sentence embeddings (768-dim) and compute continuous sentiment polarity scores: $S = P(\text{positive}) - P(\text{negative}) \in [-1.0, 1.0]$.
- **Topic Discovery**: Dynamically identifies latent discussion themes and clusters social posts using **BERTopic** (UMAP dimensionality reduction + HDBSCAN density clustering + c-TF-IDF keyword extraction).
- **Temporal Topic Tracking**: Aggregates discussion metrics over discrete time intervals (Topic Volume, Growth Rate, Weighted Engagement, Sentiment Change $\Delta S$, Hashtag Activity, Unique Users).
- **Feature Engineering & Future Ground-Truth Labeling**: Builds comprehensive temporal feature vectors (including Acceleration, Lags 1–3, Rolling 3/7 moving averages) and assigns binary ground-truth target labels (`Trending = 1` vs `0`) based on future volume and growth thresholds over forward lookahead windows.
- **Strict Time-Based Partitioning (70% - 20% - 10%)**: Chronologically partitions datasets into Train (70%), Validation (20%), and Test (10%) sets without temporal shuffling to completely prevent future data leakage.
- **Hybrid Prediction Modeling**: Combines statistical time-series forecasting (**SARIMA**) with social listening signals via **Feature Fusion**, evaluated and classified using **LightGBM** with validation-tuned optimal decision boundaries.

---

## 2. Pipeline Architecture Diagram

```mermaid
flowchart TD
    %% Stage 1
    subgraph S1["Stage 1: Data Collection & Ingestion"]
        D1["Raw Posts: Text, Timestamp, Likes, Shares, Comments, Hashtags, Followers"]
    end

    %% Stage 2
    subgraph S2["Stage 2: Preprocessing Data"]
        P1["Remove Duplicate"] --> P2["Remove Spam Bot"]
        P2 --> P3["Text Cleaning<br/><i>• Remove HTML<br/>• Normalize whitespace</i>"]
        P3 --> P4["Remove URLs"]
        P4 --> P5["Text Normalization<br/><i>• Unicode NFKC<br/>• Lowercase<br/>• Repeated-character handling</i>"]
        P5 --> P6["Preserve Semantic Information<br/><i>• Hashtags<br/>• Meaningful emojis<br/>• Multilingual negation context</i>"]
    end

    %% Stage 3
    subgraph S3["Stage 3: Multilingual Representation & Sentiment (XLM-RoBERTa)"]
        direction TB
        subgraph BranchA["Contextual Embeddings"]
            E1["XLM-RoBERTa Encoder"] --> MP["Attention Mean Pooling"]
            MP --> CSE["Contextual sentence embeddings (768-dim)"]
        end
        subgraph BranchB["Sentiment Analysis"]
            E2["XLM-RoBERTa Encoder"] --> SCH["Sentiment Classification Head<br/><i>(Pos / Neg / Neu probabilities)</i>"]
            SCH --> SS["Sentiment Score<br/><b>S = P(positive) - P(negative)</b>"]
        end
    end

    %% Stage 4
    subgraph S4["Stage 4: Topic Detection (BERTopic)"]
        T1["Embeddings"] --> T2["Clustering<br/><i>(UMAP 5D + HDBSCAN)</i>"]
        T2 --> T3["Topic Representation<br/><i>(c-TF-IDF keyword extraction)</i>"]
    end

    %% Stage 5
    subgraph S5["Stage 5: Topic Tracking"]
        TT["Topic Metrics Over Time (Daily/Hourly):<br/>• Topic Volume &nbsp;&nbsp;• Growth Rate &nbsp;&nbsp;• Weighted Engagement<br/>• Sentiment Delta (ΔS) &nbsp;&nbsp;• Hashtag Activity &nbsp;&nbsp;• Unique Users"]
    end

    %% Stage 6
    subgraph S6["Stage 6: Feature Engineering & Label Data"]
        direction LR
        FE["Feature Extraction:<br/>• Base Metrics<br/>• Acceleration<br/>• Lags (1, 2, 3)<br/>• Rolling (3, 7)"]
        LD["Label Data:<br/>• Forward Lookahead Window (k=3)<br/>• Future Volume & Future Growth<br/><b>Trending = 1 / Not Trending = 0</b>"]
    end

    %% Stage 7
    subgraph S7["Stage 7: Time-Based Data Splitting"]
        Split["Chronological Split (No Shuffling):<br/>• Train (70%)<br/>• Validation (20%)<br/>• Test (10%)"]
    end

    %% Stage 8
    subgraph S8["Stage 8: Hybrid Prediction Models & Feature Fusion"]
        subgraph Forecasting["Topic Volume Forecasting (SARIMA)"]
            SAR1["Historical Topic Volume Trajectory"] --> SAR2["SARIMA Future Volume Forecast"]
        end
        subgraph Fusion["Feature Fusion"]
            F1["Social Listening Features"] & F2["SARIMA Forecast Features"] --> F3["Combined Feature Vector"]
        end
        F3 --> LGB["Trend Prediction Classifier<br/><b>(LightGBM GBDT)</b>"]
    end

    Out["Ranked Predicted Emerging Topics<br/>+ Publication-Grade Evaluation Plots"]

    %% Connections
    D1 --> P1
    P6 --> E1
    P6 --> E2
    CSE --> T1
    SS --> TT
    T3 --> TT
    TT --> FE
    TT --> LD
    FE --> Split
    LD --> Split
    Split --> SAR1
    Split --> F1
    SAR2 --> F2
    LGB --> Out
```

---

## 3. Repository Directory Structure & File Manifest

```text
social_listening/
├── .gitignore                          # Comprehensive ignore policy (media, data, caches, models)
├── requirements.txt                    # Python dependencies with strict versions
├── README.md                           # Master architectural and operational documentation
├── main.py                             # Unified CLI runner for individual stages or full pipeline
│
├── configs/                            # Centralized configurations
│   ├── config.yaml                     # Hyperparameters, thresholds, and paths for Stages 1–8
│   └── logging_config.py               # Standardized Loguru console and rotating file logger
│
├── data/                               # Staged pipeline data storage (git-ignored, kept via .gitkeep)
│   ├── 01_raw/                         # Raw ingested social posts (social_listening_30days.csv, input_data.parquet)
│   ├── 02_preprocessed/                # Cleaned, deduplicated, and normalized posts (preprocessed_posts.parquet)
│   ├── 03_embeddings_sentiment/        # XLM-RoBERTa embeddings (.npy) and sentiment scores (.parquet)
│   ├── 04_topics/                      # BERTopic clustering assignments and topic information
│   ├── 05_topic_tracking/              # Aggregated time-series metrics per topic
│   ├── 06_features_labels/             # Tabular engineered features and future ground-truth labels
│   ├── 07_splits/                      # Chronological splits: train.parquet, valid.parquet, test.parquet
│   └── 08_predictions/                 # SARIMA forecasts, LightGBM probabilities, ranked CSV, and plots/
│
├── models/                             # Trained model weights and checkpoints (git-ignored)
│   ├── bertopic/                       # Fitted BERTopic model artifact
│   ├── sarima/                         # Fitted SARIMA time-series model parameters
│   └── lightgbm/                       # Trained LightGBM booster model (lightgbm_trend_model.joblib)
│
├── notebooks/                          # Jupyter Notebooks (Pre-configured with social_listening kernel)
│   ├── 01_data_exploration_and_preprocessing.ipynb # Stage 1 ingestion & Stage 2 step-by-step cleaning
│   ├── 02_topic_modeling_bertopic.ipynb            # Stage 3 multilingual embeddings & Stage 4 BERTopic
│   ├── 03_trend_prediction_evaluation.ipynb        # Stages 5–8 pipeline execution & visual evaluation
│   └── 04_model_evaluation_and_visualization.ipynb # Dedicated visual analytics, curves, heatmaps & dashboard
│
├── scripts/                            # Operational utility & crawler scripts
│   └── crawl_social_media_30days.py    # Live Reddit & YouTube crawler across 5 domains (30-day window)
│
└── src/                                # Modular production-grade Python package
    ├── __init__.py
    │
    ├── stage_01_data_collection/       # [STAGE 1] INGESTION & DATA VALIDATION
    │   ├── __init__.py
    │   ├── schema.py                   # Pydantic validation schema for required social media fields
    │   └── collector.py                # Schema alias mapper, Pydantic validator, and parquet persistence
    │
    ├── stage_02_preprocessing/         # [STAGE 2] TEXT PREPROCESSING & FILTERING
    │   ├── __init__.py
    │   ├── deduplicator.py             # Remove Duplicate: Drops exact and textual duplicates
    │   ├── spam_filter.py              # Remove Spam Bot: Filters accounts by followers, hashtags, and URLs
    │   ├── text_cleaner.py             # Text Cleaning: Removes HTML tags, strips URLs, normalizes whitespace
    │   ├── text_normalizer.py          # Text Normalization: NFKC Unicode, lowercase, repeated-char reduction
    │   ├── semantic_preserver.py       # Preserve Semantic: Protects hashtags, emojis, and negation context
    │   └── pipeline.py                 # Sequential orchestrator for all Stage 2 steps
    │
    ├── stage_03_multilingual_sentiment/# [STAGE 3] MULTILINGUAL EMBEDDINGS & SENTIMENT ANALYSIS
    │   ├── __init__.py
    │   ├── encoder.py                  # XLM-RoBERTa Transformer Encoder for hidden states and logits
    │   ├── pooling.py                  # Attention-mask-aware Mean Pooling for sentence embeddings
    │   ├── sentiment_head.py           # Softmax classification head & Sentiment Score computation
    │   └── pipeline.py                 # Dual-branch extractor for dense embeddings and sentiment scores
    │
    ├── stage_04_topic_detection/       # [STAGE 4] TOPIC DISCOVERY (BERTopic)
    │   ├── __init__.py
    │   ├── embeddings.py               # Ingestion and validation of precomputed sentence embeddings
    │   ├── clustering.py               # UMAP dimensionality reduction & HDBSCAN density clustering
    │   ├── representation.py           # Class-based TF-IDF (c-TF-IDF) topic keyword extraction
    │   └── bertopic_model.py           # BERTopic wrapper for fitting and topic assignment
    │
    ├── stage_05_topic_tracking/        # [STAGE 5] TEMPORAL TOPIC TRACKING
    │   ├── __init__.py
    │   ├── time_aggregator.py          # Resamples posts into discrete time buckets (Hourly / Daily)
    │   ├── metrics_calculator.py       # Computes Volume, Growth, Engagement, Sentiment Delta, Hashtags, Users
    │   └── tracker.py                  # Manages historical topic trajectory metrics
    │
    ├── stage_06_feature_engineering_labeling/ # [STAGE 6] FEATURE ENGINEERING & TREND LABELING
    │   ├── __init__.py
    │   ├── feature_extractor.py        # Extracts 7 base dimensions + Acceleration + Lags (1,2,3) + Rolling stats
    │   ├── labeler.py                  # Generates Trending (1/0) ground-truth using forward lookahead window
    │   └── dataset_builder.py          # Assembles complete feature matrix and target labels
    │
    ├── stage_07_time_based_split/      # [STAGE 7] CHRONOLOGICAL DATA PARTITIONING
    │   ├── __init__.py
    │   └── splitter.py                 # Strictly chronological split: Train (70%), Valid (20%), Test (10%)
    │
    ├── stage_08_prediction_models/     # [STAGE 8] TIME-SERIES FORECASTING & TREND PREDICTION
    │   ├── __init__.py
    │   ├── sarima_forecaster.py        # SARIMA model forecasting future topic discussion volume
    │   ├── feature_fusion.py           # Concatenates Social Listening features with SARIMA forecast features
    │   ├── lightgbm_model.py           # LightGBM binary classifier predicting trending likelihood
    │   ├── evaluator.py                # Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC
    │   ├── visualizer.py               # Generates ROC, PR, Confusion Matrix & Feature Importance plots
    │   └── pipeline.py                 # End-to-end prediction orchestrator generating ranked trending topics
    │
    └── utils/                          # SHARED UTILITIES
        ├── __init__.py
        ├── file_io.py                  # Safe file I/O for CSV, Parquet, JSON, YAML, and Pickle formats
        └── metrics_utils.py            # Mathematical utility functions (Growth Rate, Acceleration, sMAPE, MAE)
```

---

## 4. Detailed Breakdown of Pipeline Stages & Steps

### Stage 1: Data Collection & Ingestion
- Standardizes common column aliases (`text`, `timestamp`, `likes`, `shares`, `comments`, `hashtags`, `followers`, `platform`, `topic_category`).
- Validates data integrity using Pydantic's `RawPostSchema`, automatically dropping or flagging malformed rows.
- Automatically persists raw validated batches into `data/01_raw/raw_posts.parquet`.

### Stage 2: Preprocessing Data
Executes a strict 6-step sequential text processing chain:
1. **Remove Duplicate**: Discards exact post duplicates and duplicate text hashes.
2. **Remove Spam Bot**: Filters out bot accounts (under 5 followers, excessive hashtag stuffing $\ge 15$, or link spam $\ge 4$).
3. **Text Cleaning**: Strips HTML tags and normalizes erratic line breaks and whitespaces.
4. **Remove URLs**: Cleans out web URLs (`http://`, `https://`, `bit.ly`, `www.`).
5. **Text Normalization**: Applies Unicode NFKC normalization, uniform lowercasing, and collapses elongated character repetitions (e.g., `sooooo` $\rightarrow$ `soo`).
6. **Preserve Semantic Information**: Retains hashtags as contextual tokens, preserves sentiment-bearing emojis, and protects multilingual negation tokens (`not`, `no`, `never`, `cannot`, `without`, `không`, `chưa`, `chẳng`...) from aggressive stopword stripping.

### Stage 3: Multilingual Representation & Sentiment Analysis (XLM-RoBERTa)
- **Branch A (Contextual Sentence Embeddings)**: Passes normalized text through XLM-RoBERTa (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) and executes **attention-masked Mean Pooling** over token hidden states to produce dense 768-dimensional sentence vectors saved to `sentence_embeddings.npy`.
- **Branch B (Sentiment Analysis)**: Passes encoder outputs through the Sentiment Classification Head to calculate probabilities across 3 polarity classes (Positive, Negative, Neutral) and derives the continuous **Sentiment Score**:
  $$S = P(\text{positive}) - P(\text{negative}) \quad \in [-1.0, 1.0]$$

### Stage 4: Topic Detection (BERTopic)
1. **Embeddings**: Ingests contextual sentence embeddings from Stage 3.
2. **Clustering**: Non-linearly compresses embeddings using **UMAP** (5 components, cosine distance) followed by **HDBSCAN** density clustering to separate cohesive topic clusters from background noise (outliers, label `-1`).
3. **Topic Representation**: Computes **Class-based TF-IDF (c-TF-IDF)** to extract key representative terms for each discovered topic.
4. Outputs `data/04_topics/posts_with_topics.parquet` and fitted model to `models/bertopic/`.

### Stage 5: Topic Tracking Over Time
Resamples posts into discrete temporal buckets (default: `1D` daily buckets), tracking 6 critical dimensions:
- **Topic Volume ($V_t$)**: Total post count per time window.
- **Growth Rate ($\text{Growth}_t$)**: Rate of volume change relative to the preceding window.
- **Engagement**: Weighted sum of user interactions ($1.0 \times \text{Likes} + 2.0 \times \text{Shares} + 1.5 \times \text{Comments}$).
- **Sentiment Change ($\Delta S_t$)**: Shift in average sentiment polarity score ($S_t - S_{t-1}$).
- **Hashtag Activity**: Count of active distinct hashtags.
- **Unique Users**: Count of unique accounts driving the conversation.

### Stage 6: Feature Engineering & Trend Labeling
- **Feature Engineering**: Constructs feature vectors from the base tracking dimensions, extended with **Acceleration** (second derivative of Volume), multi-period lag indicators (**Lag-1, Lag-2, Lag-3**), and moving averages (**Rolling-3, Rolling-7**).
- **Label Data**: Looks ahead over a forward horizon of $k=3$ periods:
  - Computes future aggregated volume (**Future Volume**) and future growth percentage (**Future Growth**).
  - Assigns ground-truth target: `Trending = 1` if both future growth $\ge 50\%$ and future volume $\ge 5$; otherwise `Not Trending = 0`.

### Stage 7: Time-Based Data Splitting
- Arranges all topic time-bucket records chronologically.
- Divides data into: **Train (70%)**, **Validation (20%)**, and **Test (10%)**.
- **Never shuffles temporal records** to strictly safeguard against future data leakage into training models.

### Stage 8: Prediction Models & Feature Fusion
1. **Topic Volume Forecasting (SARIMA)**: Fits seasonal autoregressive time-series models (order `[1, 1, 1]`, seasonal order `[1, 1, 0, 7]`) on historical topic volume curves to forecast anticipated future volume.
2. **Feature Fusion**: Merges present Social Listening features with prospective SARIMA forecast features into a unified **Combined Feature Vector**.
3. **Trend Prediction (LightGBM)**: Trains a gradient boosting binary classifier optimized with Binary Logloss, automatically tunes the classification decision threshold on the Validation set, and evaluates on the Test set.
4. **Outputs**: Generates prioritized reports of topics with high predicted probabilities of becoming viral trends (`predicted_trending_topics.csv`) alongside evaluation metrics and publication-grade evaluation charts.

---

## 5. Core Mathematical Formulations & Algorithms

### 5.1. Attention-Masked Mean Pooling (Sentence Embeddings)
Given token hidden state vector $\vec{h}_i$ and attention mask $m_i \in \{0, 1\}$ for sequence length $L$:
$$\vec{e}_{\text{sentence}} = \frac{\sum_{i=1}^{L} (\vec{h}_i \cdot m_i)}{\sum_{i=1}^{L} m_i}$$

### 5.2. Continuous Sentiment Polarity Score
Using Softmax classification probabilities:
$$S = P(\text{positive}) - P(\text{negative}) \quad \in [-1.0, 1.0]$$

### 5.3. Class-Based TF-IDF (c-TF-IDF)
Keyword weighting for term $t$ in topic cluster $c$:
$$W_{t, c} = \|\text{tf}_{t, c}\| \times \log\left(1 + \frac{A}{\text{tf}_t}\right)$$
Where:
- $\text{tf}_{t, c}$: Frequency of word $t$ inside topic cluster $c$.
- $\text{tf}_t$: Aggregate frequency of word $t$ across all corpus documents.
- $A$: Average count of words per cluster ($\frac{1}{|C|} \sum_c \sum_t \text{tf}_{t, c}$).

### 5.4. Growth Rate & Acceleration
- **Growth Rate**:
  $$\text{Growth}_t = \frac{V_t - V_{t-1}}{\max(V_{t-1}, 1)}$$
- **Acceleration**:
  $$\text{Acceleration}_t = \text{Growth}_t - \text{Growth}_{t-1}$$

### 5.5. Weighted Engagement Score
$$\text{Engagement} = w_{\text{likes}} \cdot \text{Likes} + w_{\text{shares}} \cdot \text{Shares} + w_{\text{comments}} \cdot \text{Comments}$$

### 5.6. SARIMA Time-Series Evaluation Metrics
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
- **Symmetric Mean Absolute Percentage Error (sMAPE)**:
  $$\text{sMAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \frac{2 \cdot |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + \epsilon}$$

---

## 6. .gitignore Policy (Media & Heavy Artifact Protection)

The repository [.gitignore](file:///c:/Users/THANH%20CONG/Documents/social_listening/.gitignore) strictly enforces repository hygiene and storage constraints:
- **Blocks all image formats**: `*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.bmp`, `*.webp`, `*.tiff`, `*.svg`, `*.raw`, `*.psd`, `*.heic`, etc.
- **Blocks all video formats**: `*.mp4`, `*.avi`, `*.mov`, `*.mkv`, `*.flv`, `*.wmv`, `*.webm`, `*.m4v`, etc.
- **Blocks large datasets & artifacts**: All files under `data/*` and `models/*` (preserving empty folders via `.gitkeep`).
- **Blocks caches & virtual environments**: `venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.log`.

---

## 7. Installation & Execution Guide

### 7.1. Environment Setup

#### Step 1: Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 2: Install Package Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Register Jupyter Kernel (for Notebooks)
```bash
python -m ipykernel install --user --name social_listening --display-name "Python (social_listening venv)"
```

---

### 7.2. Live Multi-Domain Social Media Crawler (Strict 30-Day Window)

The repository includes a production-grade live online crawler in [`scripts/crawl_social_media_30days.py`](file:///c:/Users/THANH%20CONG/Documents/social_listening/scripts/crawl_social_media_30days.py). It collects 100% genuine social media discussions from Reddit feeds and verified YouTube channels strictly within the last 30 days across 5 targeted real-world topics:

1. **Technology**: Windows 11 updates, Recall, BSOD, GPU/RAM pricing (`technology_windows_hardware`)
2. **Sports**: UEFA Champions League matches, manager debates (`sports_ucl`)
3. **Gaming**: Game of the Year (GOTY) and The Game Awards discussions (`game_goty`)
4. **Artificial Intelligence**: Claude 3.5 Sonnet vs GPT-4o, OpenAI vs Anthropic debates (`ai_claude_vs_gpt`)
5. **Entertainment**: The Oscars, Academy Awards, Best Picture race (`entertainment_oscars`)

To crawl and refresh raw social discussions:
```bash
python scripts/crawl_social_media_30days.py
```

This generates and saves:
- `data/01_raw/social_listening_30days.csv` (1,513+ verified posts)
- `data/01_raw/social_listening_dataset.csv`
- `data/01_raw/input_data.parquet`

---

### 7.3. Configuration

All hyperparameters, file paths, model configurations, and thresholds are centrally controlled in [`configs/config.yaml`](file:///c:/Users/THANH%20CONG/Documents/social_listening/configs/config.yaml).

---

### 7.4. Running the Pipeline via CLI

#### Complete End-to-End Execution (Stages 1 through 8):
```bash
python main.py --all
```

#### Running Individual Stages Specifically:
```bash
# Stage 1: Data Collection & Schema Validation
# Automatically searches data/01_raw/input_data.parquet or social_listening_30days.csv
python main.py --stage 1

# Or specify a custom raw input file:
python main.py --stage 1 --input data/01_raw/social_listening_30days.csv

# Stage 2: Data Preprocessing (Deduplication, Spam Filter, Cleaning, Normalization)
python main.py --stage 2

# Stage 3: Sentence Embeddings & Sentiment Analysis (XLM-RoBERTa)
python main.py --stage 3

# Stage 4: Topic Discovery (BERTopic + UMAP + HDBSCAN + c-TF-IDF)
python main.py --stage 4

# Stage 5: Temporal Topic Tracking Over Time
python main.py --stage 5

# Stage 6: Feature Engineering & Ground-Truth Trend Labeling
python main.py --stage 6

# Stage 7: Chronological Data Splitting (70% Train, 20% Valid, 10% Test)
python main.py --stage 7

# Stage 8: SARIMA Forecasting, Feature Fusion, and LightGBM Trend Prediction
python main.py --stage 8
```

#### Output Artifacts & Inspection:
- **Ranked Predicted Trending Topics**: `data/08_predictions/predicted_trending_topics.csv`
- **Full Test Predictions**: `data/08_predictions/all_test_predictions.parquet`
- **Numerical Evaluation Metrics**: `data/08_predictions/evaluation_metrics.json`
- **Trained Model Checkpoint**: `models/lightgbm/lightgbm_trend_model.joblib`
- **Visual Evaluation Plots**: Located in `data/08_predictions/plots/`:
  - `evaluation_dashboard.png`: 4-panel master dashboard
  - `confusion_matrix.png`: Heatmap of True Positives, False Positives, True Negatives, False Negatives
  - `roc_curve.png`: Receiver Operating Characteristic curve with AUC score
  - `precision_recall_curve.png`: Precision-Recall curve with Average Precision (AP)
  - `feature_importance.png`: Feature gain bar chart (Social Listening vs SARIMA features)

---

## 8. Notebooks & Visual Analytics Guide

All notebooks in [`notebooks/`](file:///c:/Users/THANH%20CONG/Documents/social_listening/notebooks) are pre-configured to use the project kernel **`Python (social_listening venv)`**:

| Notebook | Focus & Stages | Description |
|---|---|---|
| [`01_data_exploration_and_preprocessing.ipynb`](file:///c:/Users/THANH%20CONG/Documents/social_listening/notebooks/01_data_exploration_and_preprocessing.ipynb) | **Stages 1 & 2** | Step-by-step raw data exploration, schema ingestion (`DataCollector`), deduplication (`remove_duplicates`), spam filtering (`filter_spam_and_bots`), text cleaning (`clean_text`), NFKC normalization (`normalize_text`), and semantic preservation (`preserve_semantic_information`). |
| [`02_topic_modeling_bertopic.ipynb`](file:///c:/Users/THANH%20CONG/Documents/social_listening/notebooks/02_topic_modeling_bertopic.ipynb) | **Stages 3 & 4** | Multilingual sentence representation via XLM-RoBERTa, sentiment score distribution, and BERTopic clustering analysis (UMAP + HDBSCAN + c-TF-IDF). |
| [`03_trend_prediction_evaluation.ipynb`](file:///c:/Users/THANH%20CONG/Documents/social_listening/notebooks/03_trend_prediction_evaluation.ipynb) | **Stages 5 to 8** | Topic metrics tracking over time, feature matrix construction, chronological split, SARIMA forecasting, LightGBM classification, and publication-grade evaluation curves. |
| [`04_model_evaluation_and_visualization.ipynb`](file:///c:/Users/THANH%20CONG/Documents/social_listening/notebooks/04_model_evaluation_and_visualization.ipynb) | **Evaluation Suite** | Interactive visual evaluation suite: Confusion Matrix heatmaps, ROC curve, Precision-Recall curve, Decision Threshold optimization, Feature Importance gain, Calibration curves, and SARIMA trajectory plots. |

### Selecting the Kernel in VS Code / IDE:
1. Open any `.ipynb` file in VS Code.
2. In the top-right corner of the notebook editor, click **Select Kernel** $\rightarrow$ **Python Environments...**
3. Select **`Python (social_listening venv)`** (or `./venv/Scripts/python.exe`).
4. Click **Run All** to execute interactively without dependency errors.
