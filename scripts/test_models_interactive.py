"""
Interactive Testing Script for Social Listening Models
Allows interactive testing of:
1. Sentiment Analysis & Polarity Scoring (XLM-RoBERTa)
2. Topic Detection & Keyword Representation (BERTopic + XLM-RoBERTa Embeddings)
3. Trend Prediction Classification (LightGBM)
"""

import sys
import os

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from bertopic import BERTopic
from loguru import logger
from src.stage_03_multilingual_sentiment.encoder import XLMRoBERTaEmbeddingEncoder, XLMRoBERTaSentimentClassifier
from src.stage_03_multilingual_sentiment.pooling import mean_pooling
from src.stage_03_multilingual_sentiment.sentiment_head import SentimentClassificationHead, compute_sentiment_score


def test_sentiment():
    print("\n" + "=" * 65)
    print(">>> 1. TEST SENTIMENT ANALYSIS (XLM-RoBERTa Sentiment Classifier)")
    print("=" * 65)
    
    classifier = XLMRoBERTaSentimentClassifier()
    head = SentimentClassificationHead()
    
    sample_texts = [
        "This product is amazing, the quality exceeded all my expectations!",
        "Terrible service, unhelpful staff, and an absolute disappointment.",
        "The weather is partly cloudy today with a temperature of around 25 degrees.",
        "The new AI model has impressive reasoning speed and benchmark scores.",
        "Worst update ever. Broken features and terrible performance!"
    ]
    
    print("\n[Sentiment Evaluation on Sample Posts]:")
    for text in sample_texts:
        encoded = classifier.tokenize([text])
        logits = classifier.forward(encoded)
        probs = head.predict_probabilities(logits)[0]
        score = probs[2] - probs[0]  # P(pos) - P(neg)
        label = "POSITIVE" if score > 0.1 else ("NEGATIVE" if score < -0.1 else "NEUTRAL")
        print(f"\n- Post: \"{text}\"")
        print(f"  -> Polarity: {label} | Sentiment Score S: {score:+.3f}")
        print(f"  -> Probabilities: Negative: {probs[0]:.1%} | Neutral: {probs[1]:.1%} | Positive: {probs[2]:.1%}")


def test_topic_detection():
    print("\n" + "=" * 65)
    print(">>> 2. TEST TOPIC DETECTION (BERTopic + XLM-RoBERTa Embeddings)")
    print("=" * 65)
    model_path = "models/bertopic/model"
    if not os.path.exists(model_path):
        print(f"BERTopic model checkpoint not found at: {model_path}.")
        return

    print(f"1. Loading fitted BERTopic model from: {model_path}...")
    topic_model = BERTopic.load(model_path)
    
    print("2. Initializing XLM-RoBERTa Embedding Encoder for contextual representations...")
    encoder = XLMRoBERTaEmbeddingEncoder("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    
    test_posts = [
        "Manchester City and Arsenal are competing fiercely for the Premier League title.",
        "AMD announced the new Ryzen AI processor with 192GB memory for gaming laptops.",
        "Claude Code and AI agents are changing software development workflows completely.",
        "Elden Ring new boss battle difficulty is insane but the gameplay is awesome."
    ]
    
    print("3. Extracting embeddings and mapping topics...")
    encoded = encoder.tokenize(test_posts)
    outputs = encoder.forward(encoded)
    embeds = mean_pooling(outputs["hidden_states"], outputs["attention_mask"], normalize=True)
    
    topics, probs = topic_model.transform(test_posts, embeddings=embeds)
    
    print("\n[Topic Classification Results]:")
    for post, t in zip(test_posts, topics):
        t_info = topic_model.get_topic_info(t)
        name = t_info["Name"].values[0] if not t_info.empty else f"Topic {t}"
        print(f"\n- Post: \"{post}\"")
        print(f"  -> Assigned to: Topic ID {t} | Topic Name: {name}")


def test_trend_prediction():
    print("\n" + "=" * 65)
    print(">>> 3. TEST TREND PREDICTION & DURATION ESTIMATION (LightGBM)")
    print("=" * 65)
    model_path = "models/lightgbm/lightgbm_trend_model.joblib"
    if not os.path.exists(model_path):
        print(f"LightGBM trend model checkpoint not found at: {model_path}.")
        return
        
    model = joblib.load(model_path)
    
    # 19 features:
    # ['topic_volume', 'growth_rate', 'engagement', 'avg_sentiment', 'sentiment_change',
    #  'positive_ratio', 'negative_ratio', 'neutral_ratio', 'hashtag_activity', 'hashtag_growth',
    #  'acceleration', 'unique_users', 'volume_lag_1', 'growth_rate_lag_1', 'engagement_lag_1',
    #  'rolling_mean_volume_3', 'rolling_mean_engagement_3', 'sarima_forecast_volume', 'sarima_forecast_growth']
    sample_features = np.array([
        # Scenario A: Explosive trend (Volume 85, Engagement 1,500, Growth +150%, High historical lags)
        [85.0, 1.50, 1500.0, 0.45, 0.35, 0.70, 0.05, 0.25, 12, 0.80, 2.0, 65, 35.0, 0.8, 600.0, 45.0, 800.0, 180.0, 1.10],
        # Scenario B: Low volume & stagnant engagement (Volume 1, Engagement 20, Growth -60%, Low lags)
        [1.0, -0.60, 20.0, -0.10, -0.20, 0.10, 0.50, 0.40, 0, 0.0, -1.0, 1, 1.0, -0.2, 15.0, 1.0, 20.0, 1.0, 0.0]
    ])
    
    probs = model.predict(sample_features)
    threshold = 0.23  # Tuned decision threshold from validation set
    
    from src.stage_08_prediction_models.trend_duration import estimate_trend_duration_and_persistence
    
    df_sample = pd.DataFrame(sample_features, columns=[
        'topic_volume', 'growth_rate', 'engagement', 'avg_sentiment', 'sentiment_change',
        'positive_ratio', 'negative_ratio', 'neutral_ratio', 'hashtag_activity', 'hashtag_growth',
        'acceleration', 'unique_users', 'volume_lag_1', 'growth_rate_lag_1', 'engagement_lag_1',
        'rolling_mean_volume_3', 'rolling_mean_engagement_3', 'sarima_forecast_volume', 'sarima_forecast_growth'
    ])
    df_sample["trending_probability"] = probs
    df_sample["predicted_trending"] = (probs >= threshold).astype(int)
    
    df_enriched = estimate_trend_duration_and_persistence(df_sample, threshold=threshold)
    
    scenarios = [
        "Scenario A: Explosive topic surge (Volume 85, Engagement 1,500, Growth +150%)",
        "Scenario B: Stagnant topic with low engagement (Volume 1, Engagement 20, Growth -60%)"
    ]
    
    for i, sc in enumerate(scenarios):
        p = df_enriched.loc[i, "trending_probability"]
        is_trending = df_enriched.loc[i, "predicted_trending"]
        p24 = df_enriched.loc[i, "prob_trending_24h"]
        status_24h = df_enriched.loc[i, "trend_status_24h"]
        duration = df_enriched.loc[i, "estimated_duration_days"]
        lifespan = df_enriched.loc[i, "trend_lifespan"]
        persistence = df_enriched.loc[i, "persistence_score"]
        
        print(f"\n- {sc}")
        print(f"  -> Trending Probability:    {p:.2%}")
        if is_trending:
            print(f"  -> Current Evaluation:      [TRENDING] Emerging Viral Trend (Threshold: {threshold:.2f})")
            print(f"  -> 24-Hour Horizon Status:  {status_24h} (24h Probability: {p24:.2%})")
            print(f"  -> Projected Trend Lifespan: Estimated {duration} days [{lifespan}] | Persistence: {persistence:.1f}%")
        else:
            print(f"  -> Current Evaluation:      [NOT TRENDING] Sub-threshold (Below {threshold:.2f})")
            print(f"  -> 24-Hour Horizon Status:  {status_24h} (24h Probability: {p24:.2%})")


if __name__ == "__main__":
    print("=" * 65)
    print("REAL-WORLD INTERACTIVE MODEL TESTING SUITE")
    print("=" * 65)
    print("1. Test Sentiment Analysis (XLM-RoBERTa)")
    print("2. Test Topic Detection (BERTopic)")
    print("3. Test Trend Prediction (LightGBM)")
    print("4. Run All 3 Tests")
    
    choice = input("\nSelect test (1-4) [default: 4]: ").strip() or "4"
    
    if choice == "1":
        test_sentiment()
    elif choice == "2":
        test_topic_detection()
    elif choice == "3":
        test_trend_prediction()
    elif choice == "4":
        test_trend_prediction()
        test_topic_detection()
        test_sentiment()
    else:
        print("Invalid selection.")
