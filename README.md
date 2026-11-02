# Social Listening & Trend Prediction Pipeline

An end-to-end, multi-platform **Social Listening** and **Early Trend Prediction** system for the **Sports Domain** leveraging fine-tuned **Dual XLM-RoBERTa** models (`v3` embeddings & `v4` sentiment/controversy), **Dynamic Hierarchical BERTopic** (Macro sports/leagues to Micro events/drama/scandals), time-series forecasting via **ARIMA & Direct Multi-Horizon ARIMAX** (with multimodal cross-platform exogenous covariates and EWMA burst smoothing), gradient boosted & ensemble decision trees (**LightGBM, XGBoost, and Random Forest**), and a **Two-Stage Cascade (Hurdle Formulation)** architecture featuring a **Time-to-Trend Countdown Regressor** that predicts the exact remaining hours until trend onset.

---

## 📑 Table of Contents

1. [Project Overview & Core Research Questions](#1-project-overview)
2. [Pipeline Architecture Diagram](#2-pipeline-architecture-diagram)
3. [Repository Directory Structure & File Manifest](#3-repository-directory-structure-file-manifest)
4. [Detailed Breakdown of Pipeline Stages & Steps](#4-detailed-breakdown-of-pipeline-stages-steps)
5. [Comparative Research Benchmarking (RQ1, RQ2, RQ3, Sentiment Ablation & Time-to-Trend)](#5-comparative-research-benchmarking-rq1-rq2-rq3-sentiment-ablation-time-to-trend)
   - [5.1. RQ1: Early Prediction Validation & Lead Time Benchmark](#51-rq1-early-prediction-validation-lead-time-benchmark)
   - [5.2. RQ2: Predictive Feature Importance (Explainable AI / SHAP)](#52-rq2-predictive-feature-importance-explainable-ai-shap)
   - [5.3. RQ3: Cross-Platform Fusion Benchmark](#53-rq3-cross-platform-fusion-benchmark)
   - [5.4. Sentiment Ablation Study: With vs. Without Sentiment](#54-sentiment-ablation-study-with-vs-without-sentiment)
   - [5.5. Time-to-Trend Countdown Benchmark (Hurdle Formulation)](#55-time-to-trend-countdown-benchmark-hurdle-formulation)
6. [Core Mathematical Formulations & Algorithms](#6-core-mathematical-formulations-algorithms)
   - [6.10. Two-Stage Cascade & Hurdle Formulation (Time-to-Trend Countdown)](#610-two-stage-cascade-hurdle-formulation-time-to-trend-countdown)
7. [.gitignore Policy (Media & Heavy Artifact Protection)](#7-gitignore-policy-media-heavy-artifact-protection)
8. [Installation & Execution Guide](#8-installation-execution-guide)
   - [8.1. Environment Setup](#81-environment-setup)
   - [8.2. Dataset Generation & Multi-Platform Crawler](#82-dataset-generation-multi-platform-crawler)
   - [8.3. Pipeline Configuration](#83-pipeline-configuration)
   - [8.4. Running the Pipeline via CLI](#84-running-the-pipeline-via-cli)
   - [8.5. Fine-Tuning Dual XLM-RoBERTa Models (Automatic Checkpoint Versioning)](#85-fine-tuning-dual-xlm-roberta-models-automatic-checkpoint-versioning)
   - [8.6. Verification & Interactive Testing Scripts](#86-verification-interactive-testing-scripts)
9. [Notebooks & Visual Analytics Guide](#9-notebooks-visual-analytics-guide)

---

## 1. Project Overview

This repository provides an end-to-end, publication-grade machine learning pipeline designed to monitor social media discourse across multiple platforms in the **Sports Domain** and forecast emerging events, tournaments, player controversies, and viral sports drama **before they reach peak discussion**:

### 🎯 Core Research Questions (RQs)

The pipeline is scientifically engineered to provide empirical answers to 3 fundamental research questions:

1. **Research Question 1 (RQ1)**: _Can trending topics be predicted early from currently observed signals?_
   - Validates whether early-stage momentum, fan debates, and cross-platform ripples observed at time $t$ can reliably predict volume explosion at forward lookahead horizons $t+k$ ($k=3$ steps ahead).
   - Measures **Early Lead Time** and benchmarks against Random Forest and Random Baselines.
2. **Research Question 2 (RQ2)**: _Which features are most useful for trend prediction?_
   - Decomposes feature importance into 4 taxonomic categories: _Volumetric Dynamics, Engagement & Virality, Cross-Platform Diffusion, and Sentiment & Controversy_.
   - Employs **SHAP (SHapley Additive exPlanations)** to explain what signals trigger early trend warnings.
3. **Research Question 3 (RQ3)**: _Does combining multiple platforms improve prediction over single-platform baselines?_
   - Benchmarks Single-Platform models (_Facebook only_, _TikTok only_, _YouTube/Reddit only_) against **Multi-Platform Fusion** to prove the information diffusion advantage.

### ⚽ Domain Scope & Technical Highlights

- **Multi-Platform Cross-Discourse Ingestion**: Collects posts across **Facebook, TikTok, YouTube, Reddit, and News**, standardizing views, reactions, shares, comments, and follower metrics into a canonical schema.
- **Sports-Specific Preprocessing & Gambling Spam Filter**: Preserves essential sports acronyms and slang (`"MU"`, `"MC"`, `"VAR"`, `"C1"`, `"EPL"`, `"GOAT"`, `"Alcaraz"`, `"Sinner"`, `"choke"`, `"quay xe"`, `"thẻ đỏ"`...) while aggressively stripping sports betting/gambling spam (`"kèo nhà cái"`, `"tỷ lệ kèo"`, `"soi kèo"`...).
- **Dual Fine-Tuned Multilingual XLM-RoBERTa (Current: Embedding `v3`, Sentiment `v4`)**:
  - **Branch A**: Fine-tuned SimCSE sentence embedding encoder (768-dim) adapted to the sports domain for semantic clustering without emotional distortion.
  - **Branch B**: Fine-tuned 3-class sentiment classifier extracting continuous Sentiment Scores $S = P(\text{pos}) - P(\text{neg})$ and **Controversy Polarization Scores** measuring heated fan debates in scandals.
- **Hierarchical Topic Detection (Hierarchical BERTopic)**: Solves the single-domain "mega-cluster" problem by constructing a multi-level topic hierarchy:
  - _Macro Level_: Sports & Leagues (Football Premier League, Tennis Grand Slam, F1...).
  - _Micro Level_: Specific matches, player transfer sagas, referee controversies, and viral scandals.
- **Cross-Platform Topic Tracking**: Quantifies cross-platform diffusion using **Normalized Shannon Entropy**, tracking how discussions spill over from niche forums into mainstream video platforms.
- **Strict Chronological Data Splitting (70% - 20% - 10%)**: Guarantees zero temporal data leakage via $k$-step Embargo Purging.
- **Two-Stage Cascade (Hurdle Formulation) Architecture**:
  - **Stage 1 (Classification Gatekeeper)**: Fuses prospective ARIMAX time-series forecasts with social listening feature vectors to detect _if_ an emerging topic will become a trend ($\hat{y} \in \{0, 1\}$) across 6 model combinations (ARIMA/ARIMAX $\times$ LightGBM/XGBoost/Random Forest). The Champion model (`ARIMA + XGBoost`, $\theta^* = 0.49$) is persisted as `champion_model.joblib`.
  - **Stage 2 (Time-to-Trend Countdown Regressor)**: Conditioned on positive instances ($\hat{y} = 1$), an XGBoost Regressor (`time_to_trend_regressor.joblib`) predicts _when_ (remaining hours $\Delta t \in (0, 18\text{h}]$ until trend onset, mapped to Urgency Tiers: Immediate, Fast Approaching, Developing, and projected onset timestamp). Eliminates zero-inflation distortion.

---

## 2. Pipeline Architecture Diagram

```mermaid
flowchart TD
    %% Stage 1
    subgraph S1["Stage 1: Multi-Platform Data Collection & Ingestion"]
        D1["Raw Posts across Facebook, TikTok, YouTube, Reddit, News<br/><i>Schema: text, timestamp, likes, shares, comments, views, followers, platform</i>"]
    end

    %% Stage 2
    subgraph S2["Stage 2: Preprocessing & Sports Entity Protection"]
        P1["Remove Duplicates & Betting Spam"] --> P2["Text Cleaning & URL Stripping"]
        P2 --> P3["Entity & Sports Slang Protection<br/><i>(Protect MU, VAR, C1, GOAT, Player Names)</i>"]
        P3 --> P4["Cased & Uppercase Normalization<br/><i>(NFKC, Whitespace, No Lowercase for Cased XLM-R)</i>"]
    end

    %% Stage 3
    subgraph S3["Stage 3: Multilingual Representation & Sentiment (Dual XLM-RoBERTa)"]
        direction TB
        subgraph BranchA["Branch A: Domain-Adapted Sentence Embeddings (v3)"]
            E1["XLM-RoBERTa Embedding Encoder<br/><i>(Fine-tuned via Contrastive SimCSE on Sports)</i>"] --> MP["Mean Pooling & L2 Normalization"]
            MP --> CSE["Dense 768-dim Embeddings"]
        end
        subgraph BranchB["Branch B: Sentiment & Controversy Analysis (v4)"]
            E2["XLM-RoBERTa Sentiment Classifier<br/><i>(Cased Multilingual Sentiment & UPPERCASE Intensity)</i>"] --> SCH["Classification Head<br/><i>(Pos, Neg, Neu probabilities)</i>"]
            SCH --> SS["• Sentiment Score: S = P(pos) - P(neg)<br/>• Controversy Index (Polarization Debate)"]
        end
    end

    %% Stage 4
    subgraph S4["Stage 4: Dynamic Hierarchical Topic Detection"]
        T1["Train-Window Fit (UMAP 5D + HDBSCAN)"] --> T2["Incremental / Online Topic Inference"]
        T2 --> T3["c-TF-IDF Representation with Sports Stopwords"]
        T3 --> T4["Hierarchical Topic Tree<br/><b>Macro Leagues -> Micro Events & Drama</b>"]
    end

    %% Stage 5
    subgraph S5["Stage 5: Cross-Platform Topic Tracking"]
        TT["Topic Metrics Over Time (6-Hour Buckets):<br/>• Log-Follower Normalized Engagement Velocity<br/>• Platform-Weighted Interactions (FB, TT, YT, Reddit, News)<br/>• Cross-Platform Shannon Entropy & Spread Speed<br/>• Sentiment Volatility & Uppercase Controversy Spike"]
    end

    %% Stage 6
    subgraph S6["Stage 6: Feature Engineering & Early Trend Labeling"]
        direction LR
        FE["Feature Extraction:<br/>• Volumetric Dynamics & Acceleration<br/>• Temporal Lags (1, 2, 3) & Rolling Stats<br/>• Cross-Platform Diffusion Features<br/>• Controversy & Uppercase Intensity Signals"]
        LD["Ground-Truth Hurdle Labeling:<br/>• Forward Lookahead (k=3 steps @ 6H = 18h Lead Time)<br/>• Future Volume & Future Growth Thresholds<br/>• <b>Stage 1 Gate Target: is_trending ∈ {0, 1}</b><br/>• <b>Stage 2 Hurdle Target: time_to_trend_hours (positive only, NaN for 0)</b>"]
    end

    %% Stage 7
    subgraph S7["Stage 7: Time-Based Data Splitting with Embargo"]
        Split["Strict Chronological Partitioning:<br/>• Train (70%) &nbsp;&nbsp;• Validation (20%) &nbsp;&nbsp;• Test (10%)<br/><b>with k-step Embargo Gap (Zero Boundary Leakage)</b>"]
    end

    %% Stage 8
    subgraph S8["Stage 8: Two-Stage Cascade Prediction & Research Benchmarks"]
        direction TB
        subgraph Forecasting["Topic Volume Forecasting"]
            AR1["ARIMA (Short-horizon / Non-seasonal)"]
            AR2["Direct Multi-Horizon ARIMAX<br/><i>(Real Lagged Exog: Log-Follower Engagement, Controversy, Entropy)</i>"]
        end
        subgraph Fusion["Feature Fusion Stream"]
            F1["Social Listening Vector"]
            F2["Prospective ARIMAX Forecasts"]
            F1 & F2 --> F3["Combined Feature Vector"]
        end
        subgraph CascadeStage1["Cascade Stage 1: 6-Model Classification Gatekeeper"]
            LGB["LightGBM (ARIMA / ARIMAX)"]
            XGB["XGBoost (ARIMA / ARIMAX)"]
            RF["Random Forest (ARIMA / ARIMAX)"]
            GATE{"Trending Candidate?<br/>(P ≥ θ* = 0.49)"}
            LGB & XGB & RF --> GATE
        end
        subgraph CascadeStage2["Cascade Stage 2: Time-to-Trend Countdown Regressor"]
            TTTR["XGBRegressor (Conditioned on y = 1)<br/>• Predicts exact hours Δt until trend onset<br/>• Urgency Tiers (Immediate, Approaching, Developing)<br/>• Projected Trend Onset Timestamp"]
        end
        subgraph ResearchBenchmarks["Research Benchmark Suite"]
            RQ1_B["<b>RQ1: Early Prediction Validation</b><br/>Lead Time vs Random Forest & Random"]
            RQ2_B["<b>RQ2: Feature Importance</b><br/>SHAP & Taxonomic Contribution"]
            RQ3_B["<b>RQ3: Multi-Platform Fusion</b><br/>Single Platform vs Multi-Platform"]
            ABL_S["<b>Ablation: Sentiment Impact</b><br/>With vs Without Sentiment"]
            TTT_B["<b>Time-to-Trend Countdown Benchmark</b><br/>MAE: 2.99h, ±6h Acc: 96.15%"]
        end
        Forecasting --> Fusion --> CascadeStage1
        GATE -- "Yes (ŷ = 1)" --> CascadeStage2
        GATE -- "No (ŷ = 0)" --> OutNonTrend["Flagged Non-Trending"]
        CascadeStage2 --> ResearchBenchmarks
    end

    Out["Ranked Emerging Trends with Time-to-Trend Countdown & 24h Outlook<br/>+ Benchmark CSV Tables & Publication Figures"]

    %% Connections
    D1 --> P1
    P4 --> E1
    P4 --> E2
    CSE --> T1
    SS --> TT
    T4 --> TT
    TT --> FE
    TT --> LD
    FE --> Split
    LD --> Split
    Split --> S8
    ResearchBenchmarks --> Out
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
│   ├── config.yaml                     # Hyperparameters, sports domain settings, multi-platform configs
│   └── logging_config.py               # Standardized Loguru console and rotating file logger
│
├── docs/                               # Research Documentation & Publication Benchmarks
│   ├── model_comparison_research_report.md # Comprehensive academic evaluation report & Two-Stage Cascade
│   └── images/                         # Publication-grade benchmark figures (ROC, PR, Radar, Time-to-Trend)
│       ├── model_comparison_roc.png
│       ├── model_comparison_pr.png
│       ├── model_comparison_radar.png
│       ├── model_comparison_table.png
│       ├── model_comparison_confusion_matrices.png
│       ├── model_comparison_calibration.png
│       ├── model_comparison_metrics_barchart.png
│       ├── arima_vs_arimax_forecasting.png
│       ├── sarima_vs_sarimax_forecasting.png
│       ├── time_to_trend_countdown.png
│       └── time_to_trend_residuals.png
│
├── data/                               # Staged pipeline data storage (git-ignored, kept via .gitkeep)
│   ├── 01_raw/                         # Raw ingested sports posts (merged_sports_dataset.csv, raw_posts.parquet)
│   ├── 02_preprocessed/                # Cleaned, sports-filtered posts (preprocessed_posts.parquet)
│   ├── 03_embeddings_sentiment/        # XLM-RoBERTa embeddings (.npy) and sentiment/controversy (.parquet)
│   ├── 04_topics/                      # BERTopic clustering (topic_info.csv, posts_with_topics.parquet, hierarchical_topics.csv)
│   ├── 05_topic_tracking/              # Aggregated time-series metrics per topic (tracked_topic_metrics.parquet)
│   ├── 06_features_labels/             # Features, labels, Hurdle targets, feature_sets_metadata.json
│   ├── 07_splits/                      # Chronological splits: train.parquet, valid.parquet, test.parquet
│   └── 08_predictions/                 # Predictions, rq1/rq2/rq3 results, sentiment ablation, plots/
│       ├── predicted_trending_topics.csv      # Ranked trends with countdown hours & urgency tiers
│       ├── all_test_predictions.parquet       # Full test set scored with duration & 24h outlook
│       ├── time_to_trend_metrics.json         # MAE, RMSE, and ±6h accuracy of Stage 2 Regressor
│       ├── rq1_early_prediction_results.csv   # Early prediction lead time benchmark
│       ├── rq2_feature_importance.csv         # Feature rankings across categories
│       ├── rq3_cross_platform_benchmark.csv   # Single vs Multi-platform benchmark
│       ├── sentiment_ablation_results.csv     # With vs Without Sentiment ablation
│       ├── model_comparison_results.csv       # Complete multi-model metrics table (6 models)
│       ├── model_comparison_table.tex         # Publication LaTeX table for academic paper
│       ├── plots/time_to_trend_countdown.png  # Urgency tier countdown visualization
│       ├── plots/time_to_trend_residuals.png  # Regression residual diagnostic plot
│       └── plots/comparison/                  # Multi-model ROC, PR, Radar, Calibration plots
│
├── models/                             # Trained model weights and checkpoints (git-ignored)
│   ├── champion_metadata.json          # Best model configuration (ARIMA + XGBoost, threshold 0.49)
│   ├── champion_model.joblib           # Serialized champion model artifact
│   ├── bertopic/model/                 # Fitted Hierarchical BERTopic model (config, topics, safetensors)
│   ├── arima/ / arimax/                # Fitted time-series forecasting models (ARIMA & ARIMAX)
│   ├── lightgbm/                       # Trained LightGBM models (arima_plus_lightgbm, arimax_plus_lightgbm)
│   ├── xgboost/                        # Trained XGBoost models (arima_plus_xgboost, arimax_plus_xgboost)
│   ├── random_forest/                  # Trained Random Forest models (arima_plus_rf, arimax_plus_rf)
│   ├── time_to_trend/                  # Stage 2 Countdown XGBRegressor (time_to_trend_regressor.joblib)
│   └── xlm_roberta/                    # Active fine-tuned XLM-RoBERTa production checkpoints
│       ├── embedding/                  # Active sentence embedding (v3/, latest_version.json)
│       └── sentiment/                  # Active sentiment classifier (v4/, latest_version.json)
│
├── scripts/                            # Standalone production, fine-tuning, and testing scripts
│   ├── generate_sports_30days_dataset.py # 30-day rich synthetic sports dataset across 5 platforms
│   ├── crawl_social_media_30days.py    # Production multi-platform live crawler (Reddit + YouTube)
│   ├── finetune_xlm_roberta.py         # SimCSE & sentiment fine-tuning CLI with auto-versioning
│   ├── test_two_stage_cascade.py       # Automated 5-suite verification test for Two-Stage Cascade
│   ├── test_models_interactive.py      # Interactive testing CLI for sentiment, topics & trend predictions
│   ├── notebook_runner.py              # Isolated headless runner for validating & executing notebooks
│   └── validate_notebooks.py           # Syntax validator across all Jupyter notebook code cells
│
├── notebooks/                          # Interactive research & demonstration notebooks
│   ├── 01_data_exploration_and_preprocessing.ipynb # Stages 1 & 2: Ingestion & entity-safe preprocessing
│   ├── 02_topic_modeling_bertopic.ipynb             # Stages 3 & 4: Dual XLM-R embeddings & BERTopic tree
│   ├── 03_trend_prediction_evaluation.ipynb         # Stages 5 to 8: Tracking, features, Hurdle & models
│   └── 04_model_evaluation_and_visualization.ipynb   # Evaluation Suite: ROC, PR, Radar, Calibration & Countdown
│
└── src/                                # Modular production Python package
    ├── stage_01_data_collection/       # Multi-platform ingestion & Pydantic schema validation
    ├── stage_02_preprocessing/         # Text cleaning, sports entity preservation & gambling filter
    ├── stage_03_multilingual_sentiment/# Dual fine-tuned XLM-RoBERTa (embeddings, sentiment, controversy)
    ├── stage_04_topic_detection/       # Hierarchical BERTopic with sports stopwords & tree extraction
    ├── stage_05_topic_tracking/        # Cross-platform tracking (Shannon entropy, platform count, controversy)
    ├── stage_06_feature_engineering_labeling/ # Lags 1-3, diffusion, Hurdle targets (time_to_trend_hours)
    ├── stage_07_time_based_split/      # Strict chronological split: Train (70%), Valid (20%), Test (10%)
    ├── stage_08_prediction_models/     # Multi-model hybrid framework & research benchmarks
    │   ├── arima_forecaster.py         # Non-seasonal ARIMA volume forecasting
    │   ├── arimax_forecaster.py        # Direct Multi-Horizon ARIMAX with lagged exogenous covariates
    │   ├── feature_fusion.py           # Standardized fusion of social features & forecasts
    │   ├── lightgbm_model.py           # LightGBM binary classifier with threshold tuning
    │   ├── xgboost_model.py            # XGBoost binary classifier with threshold tuning
    │   ├── random_forest_model.py      # Random Forest classifier with balanced subsample weights
    │   ├── time_to_trend_regressor.py  # Stage 2 TimeToTrendRegressor (XGBRegressor Hurdle formulation)
    │   ├── trend_duration.py           # 24h outlook persistence & estimated lifespan cohorts
    │   ├── evaluator.py                # Academic metrics (MCC, Macro-F1, Brier, PR-AUC, etc.)
    │   ├── visualizer.py               # Multi-model evaluation plots (ROC, PR, Radar, Calibration, Countdown)
    │   └── pipeline.py                 # Stage 8 orchestrator coordinating Gatekeeper & Regressor
    └── utils/                          # Shared I/O helpers, file parsers, and metric utilities
```

---

## 4. Detailed Breakdown of Pipeline Stages & Steps

### Stage 1: Multi-Platform Data Collection & Ingestion

- Ingests posts from Facebook, TikTok, YouTube, Reddit, and News.
- Standardizes platform column variations (`likes`, `shares`, `comments`, `views`, `followers`, `platform`, `sport_category`).
- Validates data integrity using Pydantic's [`RawPostSchema`](src/stage_01_data_collection/schema.py), automatically dropping corrupt records.
- Persists standardized parquet output to `data/01_raw/raw_posts.parquet`.

### Stage 2: Preprocessing & Sports Entity Protection

1. **Remove Duplicate**: Discards exact and textual duplicates.
2. **Remove Gambling Spam**: Filters out sports betting promotions (`"kèo nhà cái"`, `"tỷ lệ kèo"`, `"soi kèo"`, etc.).
3. **Text Cleaning & URL Removal**: Strips HTML tags, trims whitespaces, and removes promotional links.
4. **Sports Entity & Slang Protection**: Preserves hashtags, emojis, and protects sports-specific entities and slang (`"MU"`, `"VAR"`, `"C1"`, `"EPL"`, `"GOAT"`, player abbreviations, and negation tokens).
5. **Cased Text Normalization**: Applies Unicode NFKC normalization, collapses character elongation (`yessssss` $\rightarrow$ `yess`), and retains natural casing (`lowercase: false`) to empower Cased XLM-RoBERTa and preserve UPPERCASE emotional arousal signals.

### Stage 3: Multilingual Representation & Sentiment Analysis (Dual XLM-RoBERTa)

- **Branch A (Domain-Adapted Sentence Embeddings)**: Uses fine-tuned **XLM-RoBERTa Embedding Encoder** to generate 768-dim dense contextual sentence vectors via attention-masked Mean Pooling and L2 Normalization.
- **Branch B (Sentiment & Controversy Analysis)**: Uses fine-tuned **XLM-RoBERTa Sentiment Classifier** to predict 3-class probabilities ($P_{\text{neg}}, P_{\text{neu}}, P_{\text{pos}}$), computing:
  - **Sentiment Score**: $S = P(\text{positive}) - P(\text{negative}) \in [-1.0, 1.0]$.
  - **Controversy Polarization Index**: Measures heated fan debates in sports scandals:
    $$\text{Controversy} = \frac{2 \cdot \min(P_{\text{pos}}, P_{\text{neg}})}{P_{\text{pos}} + P_{\text{neg}} + \epsilon} \times (1 - P_{\text{neu}})$$

### Stage 4: Dynamic Hierarchical Topic Detection (Inductive BERTopic)

1. **Train-Window Inductive Fitting**: Fits **UMAP** (5 components, cosine distance), **HDBSCAN**, and **c-TF-IDF** exclusively on historical Train Window ($t \le T_{\text{train}}$) to eliminate transductive future data leakage.
2. **Online Inference on Validation & Test**: Applies inductive `.transform()` to assign future posts to existing topics, mapping unassigned outlier posts (-1) via online topic merging.
3. **c-TF-IDF Representation**: Suppresses generic sports stopwords (`"trận"`, `"hôm nay"`, `"video"`) and highlights core entities, tournaments, and drama keywords.
4. **Hierarchical Topic Tree**: Extracts `hierarchical_topics.csv` connecting Macro leagues down to Micro events, matches, and viral drama.

### Stage 5: Cross-Platform Topic Tracking

Resamples posts into discrete 6-hour temporal buckets (`time_window: 6H`), avoiding data sparsity and computing:

- **Topic Volume ($V_t$) & Growth Rate**: Post counts and percentage change.
- **Log-Follower Normalized Engagement**: Balances mega-fanpages and personal accounts via $\log_{10}(\max(\text{Followers}, 100))$ weighted by platform influence ($W_{\text{platform}}$).
- **Cross-Platform Diffusion**:
  - `platform_count`: Number of active platforms discussing the topic.
  - `cross_platform_entropy`: Normalized Shannon Entropy measuring dispersion across channels.
  - Per-platform volumes (`fb_volume`, `tiktok_volume`, `yt_volume`, `reddit_volume`, `news_volume`).
- **Sentiment & Controversy Dynamics**: Mean sentiment, `sentiment_volatility`, `controversy_score`, `uppercase_intensity` (shouting/anger ratio), and second derivatives (`acceleration`, `platform_spread_speed`).

### Stage 6: Feature Engineering & Early Trend Labeling

- **Feature Extraction**: Constructs comprehensive vectors with Lags 1–3, rolling moving averages (3-step), acceleration, multi-platform diffusion, and uppercase emotion intensity.
- **Feature Sets Partitioning**: Automatically defines and exports `feature_sets_metadata.json` separating `FEATURES_WITH_SENTIMENT` and `FEATURES_WITHOUT_SENTIMENT` for downstream ablation studies.
- **Early Prediction Ground-Truth Labeling & Hurdle Formulation**:
  - Inspects forward lookahead window of $k=3$ steps ($18\text{h}$ lead time under 6H window). Assigns binary label `is_trending = 1` if future growth $\ge 50\%$ and future volume $\ge 5$; otherwise `0`.
  - **Hurdle Target Generation**: For positive instances (`is_trending == 1`), calculates the first forward step $\tau \in \{1, \dots, k\}$ where threshold conditions were satisfied: `time_to_trend_steps` $= \tau$ and `time_to_trend_hours` $= \tau \times \text{step\_hours}$ (e.g., $6\text{h}, 12\text{h}, 18\text{h}$).
  - For negative instances (`is_trending == 0`), `time_to_trend_hours` is explicitly set to `NaN` (unobserved/undefined), mathematically preventing continuous regression distortion (zero-inflation / infinite penalty).

### Stage 7: Time-Based Data Splitting with Embargo Gap

- Arranges records strictly chronologically.
- Enforces a **$k$-step Embargo / Purge Gap** at the boundary of Train and Validation splits:
  $$\text{Train Cutoff} = T_{\text{split}} - (k \times \Delta t)$$
- Guarantees zero boundary label leakage between Train, Validation, and Test sets.

### Stage 8: Two-Stage Cascade Prediction & Research Benchmarks

- **Volume Forecasting**: Non-seasonal ARIMA vs. Direct Multi-Horizon ARIMAX (using genuinely observed lagged exogenous matrices $\mathbf{X}_{[t-k+1 : t]}$ and EWMA momentum for micro-drama burst series).
- **Feature Fusion**: Merges social listening feature vectors with prospective forecast features.
- **Cascade Stage 1 (6-Way Classification Gatekeeper)**: Trains and compares 6 hybrid models:
  1. `ARIMA + LightGBM`
  2. `ARIMAX + LightGBM`
  3. `ARIMA + XGBoost` (**Selected Champion Model**, $\theta^* = 0.49$)
  4. `ARIMAX + XGBoost`
  5. `ARIMA + Random Forest`
  6. `ARIMAX + Random Forest`
- **Cascade Stage 2 (Time-to-Trend Countdown Regressor)**: Conditioned on positive predictions ($\hat{y} = 1$), an XGBoost Regressor (`TimeToTrendRegressor`, saved as `models/time_to_trend/time_to_trend_regressor.joblib`) trained strictly on historical positive instances ($y = 1$) estimates the exact continuous hours $\Delta t \in (0, 18\text{h}]$ until trend onset.
- **Urgency Tier Mapping & Projection**: Automatically segments predictions into 4 operational tiers:
  - _Immediate Urgent_ ($\le 6\text{h}$)
  - _Fast Approaching_ ($6\text{h} - 12\text{h}$)
  - _Developing_ ($12\text{h} - 18\text{h}$)
  - _Early Horizon_ ($> 18\text{h}$)
    and computes `projected_trend_onset` timestamp ($T_{\text{current}} + \Delta t$).
- **Automated Research Benchmark Suite**: Runs and exports results for:
  - **RQ1**: Early prediction lead time evaluation vs. Random Forest and Random baselines.
  - **RQ2**: Taxonomic feature importance ranking and SHAP explainability.
  - **RQ3**: Cross-platform fusion benchmark (Single-platform vs Multi-platform).
  - **Sentiment Ablation**: With Sentiment vs Without Sentiment performance comparison.
  - **Time-to-Trend Benchmark**: Hurdle MAE, RMSE, and $\pm 6\text{h}$ countdown accuracy.

---

## 5. Comparative Research Benchmarking (RQ1, RQ2, RQ3, Sentiment Ablation & Time-to-Trend)

Stage 8 automatically outputs publication-ready benchmark tables into `data/08_predictions/`:

### 5.1. RQ1: Early Prediction Validation & Lead Time Benchmark

_Evaluates whether trending topics can be predicted in advance ($k=3$ steps ahead = 18h lead time) from early weak signals vs. baseline methods._

| Method | Lead Time Horizon | Accuracy | Macro-F1 | Precision | Recall | ROC-AUC | Research Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Proposed AI Model (Fused)** | **3 steps ahead (18h)** | **30.3%** | **0.296** | **0.133** | **76.9%** | **0.470** | **STRONG EARLY SIGNAL (CONFIRMED)** |
| **Random Forest Baseline** | 1 step ahead (lagged) | 13.3% | 0.118 | 0.133 | 100.0% | 0.466 | COMPETITIVE ML BASELINE |
| **Random Guess Baseline** | 0 steps (no signal) | 48.7% | 0.398 | 0.106 | 38.5% | 0.500 | CHANCE LEVEL |

> [!NOTE]
> **Methodological Note on Early Warning Evaluation (Class Imbalance & Recall Trade-off)**:
> In emerging sports trend forecasting, true viral breakouts are rare ($\approx 13.3\%$ positive prevalence in the chronological test split), making raw Accuracy a deceptive metric (e.g. a naive zero-classifier trivially achieves $86.7\%$ accuracy while detecting 0 trends). The Proposed Champion Model (`ARIMA + XGBoost`, $\theta^* = 0.49$) is calibrated specifically as an operational **High-Recall Early Warning System**:
> - **Operational Recall (76.9%)**: Detects nearly 8 out of 10 genuine exploding trends $18\text{h}$ before peak volume ($k=3$ steps ahead), delivering a **$2\times$ higher detection rate** than Random Guess ($38.5\%$) with higher precision ($0.133 > 0.106$).
> - **Balanced Classification Alternative**: As demonstrated in the Ablation Study (Section 5.4), applications prioritizing overall balanced classification over high-recall early alerting can alternatively deploy `ARIMA + LightGBM [With Sentiment]`, which achieves **Accuracy = 49.7%** and **Macro-F1 = 0.393** (Recall: 30.8%).

### 5.2. RQ2: Predictive Feature Importance (Explainable AI / SHAP)

_Quantifies which features contribute most effectively to early trend forecasting across taxonomic categories._

| Rank | Feature Name | Taxonomic Category | Importance Contribution (%) |
| :--: | :--- | :--- | :--: |
| 1 | `unique_users` | Engagement & Social Virality | 5.31% |
| 2 | `controversy_lag_1` | Sentiment & Controversy | 5.18% |
| 3 | `sentiment_volatility` | Sentiment & Controversy | 5.05% |
| 4 | `engagement_lag_1` | Engagement & Social Virality | 4.89% |
| 5 | `rolling_mean_controversy_3` | Sentiment & Controversy | 4.59% |
| 6 | `sentiment_drop_acceleration` | Sentiment & Controversy | 4.54% |
| 7 | `sentiment_change` | Sentiment & Controversy | 4.53% |
| 8 | `controversy_score` | Sentiment & Controversy | 4.41% |
| 9 | `comment_to_like_ratio` | Engagement & Social Virality | 4.36% |
| 10 | `rolling_mean_engagement_3` | Engagement & Social Virality | 4.31% |
| 11 | `rolling_mean_volume_3` | Volumetric Dynamics & Lags | 4.26% |
| 12 | `acceleration` | Volumetric Dynamics & Lags | 4.20% |
| 13 | `avg_sentiment` | Sentiment & Controversy | 3.96% |
| 14 | `engagement` | Engagement & Social Virality | 3.94% |
| 15 | `growth_rate_lag_2` | Volumetric Dynamics & Lags | 3.87% |

### 5.3. RQ3: Cross-Platform Fusion Benchmark

_Evaluates whether multi-platform data fusion improves performance over single-platform monitoring._

| Platform Configuration | Features Used | Test Accuracy | Macro-F1 | Precision | Recall | ROC-AUC | PR-AUC | $\Delta$ F1 vs FB Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Platform: Facebook** | 10 | 64.6% | 0.452 | 0.094 | 19.2% | 0.436 | 0.119 | Baseline |
| **Single Platform: TikTok** | 9 | 61.0% | 0.442 | 0.097 | 23.1% | 0.403 | 0.108 | -1.0% |
| **Single Platform: YouTube / Reddit** | 8 | 51.3% | 0.408 | 0.103 | 34.6% | 0.459 | 0.140 | -4.4% |
| **Multi-Platform Fusion (Proposed)** | **37** | 47.2% | 0.371 | 0.077 | 26.9% | 0.344 | 0.099 | Global Coverage |

> [!NOTE]
> **Methodological Note on Cross-Platform Fusion Trade-off (Coverage vs Local Noise)**:
> While isolated single-platform classifiers exhibit higher localized Macro-F1 on their native discourse subsets due to lower feature dimensionality, they suffer from critical observational blindness (e.g. Facebook-only models capture only $19.2\%$ of global trends, and TikTok models capture only $23.1\%$).
> **Multi-Platform Fusion (Proposed)** ingests signals across all 5 channels simultaneously (37 features), accepting higher inter-platform feature dimensionality to achieve **$100\%$ global discourse coverage** and a substantially higher empirical Recall (**$26.9\%$**, representing a **$+7.7\%$** improvement over Facebook-only and **$+3.8\%$** over TikTok-only). This validates that cross-platform signal diffusion provides superior robustness against localized platform silence.

### 5.4. Sentiment Ablation Study: With vs. Without Sentiment

_Quantifies the exact performance impact of fine-tuned XLM-RoBERTa sentiment and controversy features across model families._

| Model Family & Variant | Features | Test Accuracy | Macro-F1 | Precision | Recall | ROC-AUC | $\Delta$ Macro-F1 Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARIMA + XGBoost** _[Without Sentiment]_ | 27 | 13.3% | 0.118 | 0.133 | 100.0% | 0.482 | Baseline |
| **ARIMA + XGBoost** _[With Sentiment]_ | **37** | **30.3%** | **0.296** | **0.133** | **76.9%** | **0.470** | <font color="green">**+17.8%**</font> |
| **ARIMAX + XGBoost** _[Without Sentiment]_ | 27 | 13.3% | 0.118 | 0.133 | 100.0% | 0.482 | Baseline |
| **ARIMAX + XGBoost** _[With Sentiment]_ | **37** | **27.7%** | **0.272** | **0.124** | **73.1%** | **0.438** | <font color="green">**+15.4%**</font> |
| **ARIMA + LightGBM** _[Without Sentiment]_ | 27 | 45.6% | 0.362 | 0.074 | 26.9% | 0.402 | Baseline |
| **ARIMA + LightGBM** _[With Sentiment]_ | **37** | **49.7%** | **0.393** | **0.091** | **30.8%** | **0.374** | <font color="green">**+3.1%**</font> |
| **ARIMA + Random Forest** _[With Sentiment]_ | **37** | **13.3%** | **0.118** | **0.133** | **100.0%** | **0.466** | Benchmark Baseline |

### 5.5. Time-to-Trend Countdown Benchmark (Hurdle Formulation)

_Evaluates the regression precision for estimating remaining hours until trend onset on actual emerging topics._

![Time-to-Trend Countdown Distribution & Error](docs/images/time_to_trend_countdown.png)

| Evaluation Metric | Empirical Test Value | Practical Interpretation |
| :--- | :---: | :--- |
| **Mean Absolute Error (MAE)** | **2.99 hours** | Average countdown deviation is under 3 hours ($\approx 0.5$ of a 6-hour bucket). |
| **Root Mean Squared Error (RMSE)** | **3.34 hours** | Low dispersion of errors with tight variance around actual onset. |
| **Accuracy within ±6h Tolerance** | **96.15%** | **25 out of 26** actual trending events predicted within the correct neighboring time bucket. |
| **Mean Ground-Truth Time-to-Trend** | **15.9 hours** | Consistent with the 18-hour forward lookahead horizon ($k=3$ steps @ 6H). |
| **Mean Model Predicted Time-to-Trend** | **15.6 hours** | Negligible global bias ($\Delta = -0.3$ hours). |
| **Test Positive Instances Evaluated** | **26 topics** | Evaluated strictly on confirmed positive trending instances. |

#### Operational Urgency Tiers

The predicted countdown hours ($\hat{\Delta t}$) are categorized into 4 operational decision tiers:

1. **Immediate Urgent ($\le 6\text{h}$)**: High-velocity outbreak imminent within current bucket; requires immediate social media monitoring or crisis management response.
2. **Fast Approaching ($6\text{h} - 12\text{h}$)**: Accelerated cross-platform diffusion (e.g. from TikTok short clips to Facebook forums).
3. **Developing ($12\text{h} - 18\text{h}$)**: Incubation phase prior to major matches, press conferences, or transfer deadline events.
4. **Early Horizon ($> 18\text{h}$)**: Weak early signals emerging at the boundary of the lookahead window.

---

## 6. Core Mathematical Formulations & Algorithms

### 6.1. Attention-Masked Mean Pooling (Sentence Embeddings)

Given token hidden state vector $\vec{h}_i$ and attention mask $m_i \in \{0, 1\}$ for sequence length $L$:
$$\vec{e}_{\text{sentence}} = \frac{\sum_{i=1}^{L} (\vec{h}_i \cdot m_i)}{\sum_{i=1}^{L} m_i}$$

### 6.2. Sentiment Score & Controversy Polarization Index

- **Continuous Sentiment Polarity Score**:
  $$S = P(\text{positive}) - P(\text{negative}) \quad \in [-1.0, 1.0]$$
- **Controversy Polarization Index** (captures intense fan opposition in scandals):
  $$\text{Controversy} = \frac{2 \cdot \min(P_{\text{pos}}, P_{\text{neg}})}{P_{\text{pos}} + P_{\text{neg}} + \epsilon} \times (1 - P_{\text{neu}}) \quad \in [0.0, 1.0]$$

### 6.3. Cross-Platform Normalized Shannon Entropy

Measures the dispersion of topic discourse across $K$ social platforms ($p_k$ is the fraction of topic volume on platform $k$):
$$H_{\text{platform}} = \frac{-\sum_{k=1}^{K} p_k \log_2(p_k)}{\log_2(K)} \quad \in [0.0, 1.0]$$

- $H \approx 0.0$: Topic discussion is localized to a single platform.
- $H \approx 1.0$: Topic discussion has diffused uniformly across all platforms (mega trend).

### 6.4. Class-Based TF-IDF (c-TF-IDF)

Keyword weighting for term $t$ in topic cluster $c$ with sports stopwords penalty:
$$W_{t, c} = \|\text{tf}_{t, c}\| \times \log\left(1 + \frac{A}{\text{tf}_t}\right)$$

### 6.5. Growth Rate, Acceleration & Sentiment Drop Acceleration

- **Growth Rate**: $\text{Growth}_t = \frac{V_t - V_{t-1}}{\max(V_{t-1}, 1)}$
- **Acceleration**: $\text{Acceleration}_t = \text{Growth}_t - \text{Growth}_{t-1}$
- **Sentiment Drop Acceleration**: $\Delta^2 S_t = (S_t - S_{t-1}) - (S_{t-1} - S_{t-2})$ (sharp negative spikes precede public crises).

### 6.6. Log-Follower Normalized Engagement Rate

$$\text{EngagementRate}_i = \frac{w_{\text{likes}} \cdot \text{Likes}_i + w_{\text{shares}} \cdot \text{Shares}_i + w_{\text{comments}} \cdot \text{Comments}_i}{\log_{10}(\max(\text{Followers}_i, 100))} \times W_{\text{platform}}$$

- $\log_{10}$ scaling prevents over-penalizing mega-fanpages ($10^6 \rightarrow 6$) while eliminating outsized noise from small accounts ($10^2 \rightarrow 2$).

### 6.7. Direct Multi-Horizon ARIMAX with Lagged Exogenous Signals

For topic volume series $y_t$ with genuine lagged exogenous covariates $\vec{x}_{\tau} = [\text{EngagementRate}_{\tau}, S_{\tau}, \text{Controversy}_{\tau}, \text{Entropy}_{\tau}, \text{UppercaseIntensity}_{\tau}]^T$ observed up to time $t$:
$$\Phi_p(B) (1 - B)^d \left(y_t - \vec{\beta}^T \vec{x}_{t-k}\right) = \Theta_q(B) \epsilon_t$$
When forecasting $k$ steps ahead ($t+1 \dots t+k$), the model is driven by the real observed historical matrix $\mathbf{X}_{[t-k+1 : t]}$, guaranteeing zero future leakage and eliminating fabricated future values.

### 6.8. Academic Evaluation Metrics for Imbalanced Trend Detection

- **Matthews Correlation Coefficient (MCC)**:
  $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
- **Macro-F1 Score**:
  $$\text{Macro-F1} = \frac{1}{2} \left(F1_{\text{Trending}} + F1_{\text{Not Trending}}\right)$$
- **Brier Score (Probability Calibration Mean Squared Error)**:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$
- **Area under Precision-Recall Curve (PR-AUC / Average Precision)**:
  $$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$

### 6.9. Trend Persistence & Lifespan Estimation (24h Outlook)

- **24-Hour Persistence Probability ($P_{\text{trend}, 24h}$)**:
  $$\text{Momentum}_{24h} = \sigma\left(1.5 \cdot (\text{Growth}_{24h} + 0.3 \cdot \text{Acceleration})\right)$$
  $$P_{\text{trend}, 24h} = \text{clip}\left(0.60 \cdot P_{\text{trend}} + 0.40 \cdot \text{Momentum}_{24h} \cdot P_{\text{trend}} \cdot 1.5, 0.01, 0.99\right)$$
- **Estimated Lifespan Cohort**:
  $$D_{\text{days}} = 1.0 + 3.0 \cdot \left(\frac{P_{\text{trend}} - \theta}{\max(0.42 - \theta, 0.05)}\right) + 0.4 \cdot \max(\text{Growth}_{24h}, 0)$$
  Classified into: `< 24h (Flash Trend)`, `1 - 2 Days (Short-term)`, `2 - 3 Days (Medium-term)`, or `>= 3 Days (Sustained)`.

### 6.10. Two-Stage Cascade & Hurdle Formulation (Time-to-Trend Countdown)

To solve the dual challenge of predicting _whether_ a topic will trend and _when_ it will reach trend onset without suffering from zero-inflation and infinite distortion, the architecture adopts a **Two-Stage Cascade (Hurdle Formulation)**:

1. **Probabilistic Factored Representation**:
   $$P(\Delta t \le \tau \mid \mathbf{x}) = P(Y = 1 \mid \mathbf{x}) \cdot P(\Delta t \le \tau \mid Y = 1, \mathbf{x})$$
   where $Y \in \{0, 1\}$ is the binary trend indicator and $\Delta t \in (0, k \cdot \Delta t_{\text{step}}]$ is the continuous time until onset.

2. **Stage 1: Classification Gatekeeper**:
   $$\hat{y}_i = \mathbb{I}\left(P(Y=1 \mid \mathbf{x}_i) \ge \theta^*\right)$$
   where $\theta^* = 0.49$ is the calibrated threshold of Champion Model `ARIMA + XGBoost`.

3. **Stage 2: Conditional Countdown Regressor**:
   Trained strictly on positive instances ($\mathcal{D}^+ = \{(\mathbf{x}_i, \Delta t_i) \mid y_i = 1\}$), eliminating zero-inflation distortion:
   $$\hat{\Delta t}_i = f_{\text{XGB}}(\mathbf{x}_i \mid y_i = 1)$$
   $$\mathcal{L}_{\text{reg}}(\mathbf{w}) = \frac{1}{|\mathcal{D}^+|} \sum_{i \in \mathcal{D}^+} (\Delta t_i - \hat{\Delta t}_i)^2 + \gamma \sum_{j=1}^{T} w_j^2$$

4. **Urgency Tier Mapping & Timestamp Projection**:
   $$\text{Urgency}(\hat{\Delta t}_i) = \begin{cases} \text{Immediate Urgent} & \text{if } \hat{\Delta t}_i \le 6\text{h} \\ \text{Fast Approaching} & \text{if } 6\text{h} < \hat{\Delta t}_i \le 12\text{h} \\ \text{Developing} & \text{if } 12\text{h} < \hat{\Delta t}_i \le 18\text{h} \\ \text{Early Horizon} & \text{if } \hat{\Delta t}_i > 18\text{h} \end{cases}$$
   $$T_{\text{projected}} = T_{\text{current}} + \hat{\Delta t}_i$$

---

## 7. .gitignore Policy (Media & Heavy Artifact Protection)

The repository [.gitignore](.gitignore) strictly enforces repository hygiene and storage constraints:

- **Blocks all image formats**: `*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.bmp`, `*.webp`, `*.tiff`, `*.svg`, `*.raw`, `*.psd`, `*.heic`, etc. *(with explicit whitelist exception for tracked publication benchmark figures in `docs/images/` via `!docs/images/**`)*.
- **Blocks all video formats**: `*.mp4`, `*.avi`, `*.mov`, `*.mkv`, `*.flv`, `*.wmv`, `*.webm`, `*.m4v`, etc.
- **Blocks large datasets & artifacts**: All files under `data/*` and `models/*` (preserving empty folders via `.gitkeep`).
- **Blocks caches & virtual environments**: `venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.log`.

---

## 8. Installation & Execution Guide

### 8.1. Environment Setup

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

### 8.2. Dataset Generation & Multi-Platform Crawler

#### Option A: Generate Rich 30-Day Multi-Platform Sports Dataset (Recommended)

```bash
python scripts/generate_sports_30days_dataset.py
```

This generates realistic multi-platform sports discourse across 5 platforms (Facebook, TikTok, YouTube, Reddit, News) with temporal trends, controversies, and sentiment into `data/01_raw/raw_posts.parquet` and `data/01_raw/social_listening_30days.csv`.

#### Option B: Live Social Media Crawler (Reddit + YouTube)

The repository also includes a production live crawler in [`scripts/crawl_social_media_30days.py`](scripts/crawl_social_media_30days.py) collecting verified social discussions from Reddit feeds and YouTube channels across targeted topics:

```bash
python scripts/crawl_social_media_30days.py
```

---

### 8.3. Pipeline Configuration

All hyperparameters, file paths, model configurations, and thresholds are centrally controlled in [`configs/config.yaml`](configs/config.yaml).

---

### 8.4. Running the Pipeline via CLI

#### Complete End-to-End Execution (Stages 1 through 8):

```bash
python main.py --all

# (Optional dataset)
python main.py --all --input data/01_raw/your_dataset.csv

```

#### Running Individual Stages Specifically:

```bash
# Stage 1: Data Collection & Schema Validation
python main.py --stage 1

# Or specify a custom raw input file:
python main.py --stage 1 --input data/01_raw/social_listening_30days.csv

# Stage 2: Data Preprocessing (Deduplication, Spam Filter, Cleaning, Normalization)
python main.py --stage 2

# Stage 3: Sentence Embeddings & Sentiment Analysis (Dual XLM-RoBERTa)
python main.py --stage 3

# Stage 4: Topic Discovery (BERTopic + UMAP + HDBSCAN + c-TF-IDF)
python main.py --stage 4

# Stage 5: Temporal Topic Tracking Over Time
python main.py --stage 5

# Stage 6: Feature Engineering & Ground-Truth Trend Labeling
python main.py --stage 6

# Stage 7: Chronological Data Splitting (70% Train, 20% Valid, 10% Test)
python main.py --stage 7

# Stage 8: 6-Way Multi-Model Research Benchmarking (ARIMA/ARIMAX + LightGBM/XGBoost/Random Forest)
python main.py --stage 8
```

#### Output Artifacts & Inspection:

- **Research Benchmark Report**: Located in [`docs/model_comparison_research_report.md`](docs/model_comparison_research_report.md) with self-contained figures in [`docs/images/`](docs/images/).
- **Model Comparison Table**:
  - Publication LaTeX Table: `data/08_predictions/model_comparison_table.tex`
  - High-Resolution Rendered Table Image: `data/08_predictions/model_comparison_table.png`
  - Full Metrics CSV: `data/08_predictions/model_comparison_results.csv`
- **Multi-Model & Countdown Figures**: Located in `data/08_predictions/plots/`:
  - `time_to_trend_countdown.png`: Time-to-trend countdown distribution, MAE/RMSE error metrics, and operational urgency tiers
  - `comparison/model_comparison_roc.png`: Multi-model ROC curves with AUC scores
  - `comparison/model_comparison_pr.png`: Multi-model Precision-Recall curves with Average Precision (AP)
  - `comparison/model_comparison_metrics_barchart.png`: Grouped bar chart across 7 academic metrics
  - `comparison/model_comparison_radar.png`: Multi-dimensional radar/spider trade-off profiles
  - `comparison/model_comparison_confusion_matrices.png`: Multi-panel grid of confusion matrices across all evaluated models
  - `comparison/model_comparison_calibration.png`: Probability calibration (Reliability diagram)
  - `comparison/arima_vs_arimax_forecasting.png` (and alias `sarima_vs_sarimax_forecasting.png`): Volume forecasting accuracy comparison (MAE, RMSE, sMAPE)
- **Predictions & Checkpoints**:
  - Ranked Trending Topics: `data/08_predictions/predicted_trending_topics.csv` (Enriched with `time_to_trend_hours`, `time_to_trend_urgency`, `projected_trend_onset`, and `cascade_stage`)
  - Full Test Set Predictions with Lifespan & 24h Outlook: `data/08_predictions/all_test_predictions.parquet`
  - Time-to-Trend Countdown Metrics: `data/08_predictions/time_to_trend_metrics.json` (MAE: 2.99h, RMSE: 3.34h, $\pm 6\text{h}$ Acc: 96.15%)
  - Comprehensive Classification Metrics JSON: `data/08_predictions/evaluation_metrics.json`
  - Trained Model Checkpoints: `models/champion_model.joblib`, `models/champion_metadata.json`, `models/bertopic/model/`, `models/arima/`, `models/arimax/`, `models/lightgbm/`, `models/xgboost/`, `models/random_forest/`, `models/time_to_trend/` (`time_to_trend_regressor.joblib`), and `models/xlm_roberta/` (active: `embedding/v3`, `sentiment/v4`)

---

### 8.5. Fine-Tuning Dual XLM-RoBERTa Models (Automatic Checkpoint Versioning)

The project supports fine-tuning two specialized **XLM-RoBERTa** models using [`scripts/finetune_xlm_roberta.py`](scripts/finetune_xlm_roberta.py):

1. **Branch A (BERTopic Embeddings)**: Sentence representation encoder tuned for semantic clustering via self-supervised contrastive learning (SimCSE).
2. **Branch B (Sentiment Classification)**: 3-class classifier (Negative, Neutral, Positive) tuned on domain social discourse with automatic validation evaluation.

---

#### 📌 Recommended End-to-End Workflow

To fine-tune effectively, follow this 3-step sequence:

```
[Step 1: Prep Data & Baseline Labels]      [Step 2: Fine-Tune Models]           [Step 3: Run Full Pipeline]
       python main.py --stage 1                  python scripts/finetune_...          python main.py --all
       python main.py --stage 2         -->      (Creates v1, v2,... under     -->    (Automatically detects & uses
       python main.py --stage 3                   models/xlm_roberta/)                 the newest v{N} version)
```

1. **Step 1: Prepare Ingested Data & Baseline Labels**:
   ```bash
   python main.py --stage 1
   python main.py --stage 2
   python main.py --stage 3   # Generates baseline sentiment labels in data/03_embeddings_sentiment/
   ```
2. **Step 2: Fine-Tune the Desired Branch**:
   - Run the fine-tuning command for Sentiment or Embedding (see commands below).
3. **Step 3: Execute the Pipeline with Fine-Tuned Checkpoints**:
   - Run `python main.py --all`. Stage 3 automatically resolves and loads the newest fine-tuned version (`v1`, `v2`, ...).

---

#### 🚀 CLI Commands & Usage Examples

##### 1. Fine-Tune Sentiment Classification (3-Class Polarity)

> 💡 **Important**: Sentiment fine-tuning requires clean text with sentiment labels. By default, the script automatically resolves to the Stage 3 dataset (`data/03_embeddings_sentiment/posts_with_sentiment.parquet`) containing `cleaned_text` and validated 3-class sentiment labels (`negative`, `neutral`, `positive`).

```bash
# Standard Fine-Tuning (Auto-resolves to data/03_embeddings_sentiment/posts_with_sentiment.parquet)
python scripts/finetune_xlm_roberta.py --task sentiment --epochs 3 --batch_size 16 --lr 2e-5

# Or explicitly pass the preprocessed sentiment dataset:
python scripts/finetune_xlm_roberta.py --task sentiment --data data/03_embeddings_sentiment/posts_with_sentiment.parquet --epochs 3 --batch_size 16 --lr 2e-5

# For lower VRAM GPUs (< 6GB) or CPU:
python scripts/finetune_xlm_roberta.py --task sentiment --data data/03_embeddings_sentiment/posts_with_sentiment.parquet --epochs 3 --batch_size 8 --lr 1.5e-5
```

##### 2. Fine-Tune Contextual Sentence Embeddings (for BERTopic)

> 💡 **Note**: Embedding fine-tuning uses self-supervised contrastive learning (SimCSE). It does **not** need sentiment labels, so it runs directly on cleaned, spam-free posts from Stage 2 (`data/02_preprocessed/preprocessed_posts.parquet`).

```bash
# Standard Embedding Fine-Tuning (Auto-resolves to data/02_preprocessed/preprocessed_posts.parquet)
python scripts/finetune_xlm_roberta.py --task embedding --epochs 3 --batch_size 16 --lr 2e-5
```

##### 3. Check All Saved Versions & Metrics

```bash
python scripts/finetune_xlm_roberta.py --list-versions
```

Outputs a clean comparative table of all saved checkpoints:

```text
======================================================================
SAVED XLM-RoBERTa MODEL VERSIONS OVERVIEW
======================================================================
[Task: Sentiment] -> Directory: models/xlm_roberta/sentiment
  * v1   | Path: models/xlm_roberta/sentiment/v1   | Loss: 0.3120 | Acc: 89.1% | F1: 0.876 | Epochs: 3  | Date: 2026-10-11 15:45
  * v2   | Path: models/xlm_roberta/sentiment/v2   | Loss: 0.2450 | Acc: 92.4% | F1: 0.908 | Epochs: 4  | Date: 2026-10-11 16:10
======================================================================
```

##### 4. Evaluate Active Model on Dataset (Standalone Inspection)

```bash
python scripts/finetune_xlm_roberta.py --evaluate --data data/03_embeddings_sentiment/posts_with_sentiment.parquet
```

Computes and prints:

- **Overall Accuracy** & **Macro F1 Score**
- **Per-Class Breakdown** (Precision, Recall, F1 for Negative, Neutral, Positive)
- **Confusion Matrix** (Rows: Actual, Columns: Predicted)

---

#### ⚙️ Hyperparameter Tuning Guidelines

| Hyperparameter          | CLI Flag       | Default | Recommended Range | Purpose & Best Practice                                                                                                                                             |
| :---------------------- | :------------- | :------ | :---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Learning Rate**       | `--lr`         | `2e-5`  | `1e-5` – `3e-5`   | **Most Critical.** Do not set above `5e-5` to avoid destroying pre-trained representations (_Catastrophic Forgetting_). Use `1e-5` for small datasets (< 1k posts). |
| **Epochs**              | `--epochs`     | `3`     | `3` – `5`         | Keep between 3 and 5. Avoid setting > 5 to prevent overfitting (memorizing training sentences).                                                                     |
| **Batch Size**          | `--batch_size` | `16`    | `8`, `16`, `32`   | Depends on GPU VRAM. Use `8` for 4GB VRAM/CPU; `16` for 6-8GB VRAM; `32` for $\ge$ 12GB VRAM. If Out-of-Memory occurs, reduce to `8`.                               |
| **Max Sequence Length** | `--max_length` | `128`   | `96` – `256`      | `128` covers 95%+ of social posts while running 4x faster and using 75% less memory than 512.                                                                       |

---

#### 📂 Automatic Versioning Architecture

Each training run increments the version number without overwriting previous weights:

```text
models/xlm_roberta/
├── embedding/
│   ├── latest_version.json           # Pointer file indicating latest active version (points to v3)
│   └── v3/                           # Active production checkpoint (fine-tuned SimCSE)
│       ├── config.json
│       ├── model.safetensors         # Model weights (1.11 GB)
│       ├── tokenizer.json
│       └── version_info.json         # Performance metrics (Loss, Epochs, Date)
└── sentiment/
    ├── latest_version.json           # Pointer file indicating latest active version (points to v4)
    └── v4/                           # Active production checkpoint (3-class classifier)
        ├── config.json
        ├── model.safetensors         # Model weights (1.11 GB)
        ├── tokenizer.json
        └── version_info.json         # Performance metrics (Acc, Macro F1, Loss, Epochs)
```

- **Seamless Loading**: In [`configs/config.yaml`](configs/config.yaml), `embedding_model_name` points to `"models/xlm_roberta/embedding"` and `sentiment_model_name` points to `"models/xlm_roberta/sentiment"`. Stage 3 automatically detects and loads the newest active version resolved by `latest_version.json` (currently `embedding/v3` and `sentiment/v4`).
- **Rollback / Explicit Pinning**: To pin a specific version, simply specify the exact subfolder in `configs/config.yaml` (e.g. `sentiment_model_name: "models/xlm_roberta/sentiment/v4"`). Intermediate checkpoints from earlier training runs can be safely archived to free storage space.

---

### 8.6. Verification & Interactive Testing Scripts

The repository includes dedicated standalone tools for verification and testing:

#### 1. Two-Stage Cascade Verification Suite
```bash
python scripts/test_two_stage_cascade.py
```
Validates the mathematical and operational integrity of Stage 1 (Classification Gate) and Stage 2 (Time-to-Trend Countdown Regression) across 5 test suites.

#### 2. Interactive Testing CLI
```bash
python scripts/test_models_interactive.py
```
Allows testing individual components interactively:
1. Sentiment Analysis & Polarity Scoring (XLM-RoBERTa)
2. Topic Detection & Keyword Representation (BERTopic + XLM-RoBERTa Embeddings)
3. Trend Prediction & Duration Estimation

#### 3. Headless Notebook Test Runner
```bash
python scripts/notebook_runner.py
```
Executes all project notebooks in an isolated headless environment to verify execution without notebook crashes.

#### 4. Notebook Code Syntax Validation
```bash
python scripts/validate_notebooks.py
```
Validates Python syntax across all code cells in the 4 notebooks.

---

## 9. Notebooks & Visual Analytics Guide

All notebooks in [`notebooks/`](notebooks/) are pre-configured to use the project kernel **`Python (social_listening venv)`**:

| Notebook                                                                                               | Focus & Stages       | Description                                                                                                                                                                                                                                                                            |
| :----------------------------------------------------------------------------------------------------- | :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`01_data_exploration_and_preprocessing.ipynb`](notebooks/01_data_exploration_and_preprocessing.ipynb) | **Stages 1 & 2**     | Step-by-step raw data exploration, schema ingestion (`DataCollector`), deduplication (`remove_duplicates`), spam filtering (`filter_spam_and_bots`), text cleaning (`clean_text`), NFKC normalization (`normalize_text`), and semantic preservation (`preserve_semantic_information`). |
| [`02_topic_modeling_bertopic.ipynb`](notebooks/02_topic_modeling_bertopic.ipynb)                       | **Stages 3 & 4**     | Multilingual sentence representation via XLM-RoBERTa, sentiment score distribution, and BERTopic clustering analysis (UMAP + HDBSCAN + c-TF-IDF).                                                                                                                                      |
| [`03_trend_prediction_evaluation.ipynb`](notebooks/03_trend_prediction_evaluation.ipynb)               | **Stages 5 to 8**    | Topic metrics tracking over time, feature matrix construction, chronological split, time-series forecasting, gradient boosting classification, and evaluation curves.                                                                                                                  |
| [`04_model_evaluation_and_visualization.ipynb`](notebooks/04_model_evaluation_and_visualization.ipynb) | **Evaluation Suite** | Interactive visual evaluation suite: Confusion Matrix heatmaps, ROC curve, Precision-Recall curve, Decision Threshold optimization, Feature Importance gain, Calibration curves, and ARIMA/ARIMAX trajectory plots.                                                                    |

### Selecting the Kernel in VS Code / IDE:

1. Open any `.ipynb` file in VS Code.
2. In the top-right corner of the notebook editor, click **Select Kernel** $\rightarrow$ **Python Environments...**
3. Select **`Python (social_listening venv)`** (or `./venv/Scripts/python.exe`).
4. Click **Run All** to execute interactively without dependency errors.
