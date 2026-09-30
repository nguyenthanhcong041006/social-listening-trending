import argparse
import sys
from loguru import logger

from configs.logging_config import setup_logger
from src.utils.file_io import load_yaml

def parse_args():
    parser = argparse.ArgumentParser(
        description="Social Listening & Trend Prediction Pipeline Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=range(1, 9),
        default=None,
        help="Run an individual stage specifically (1 to 8)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the complete end-to-end pipeline from Stage 1 through Stage 8"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Custom raw input dataset path for Stage 1"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logger()
    
    config = load_yaml(args.config)
    paths = config.get("paths", {})
    
    logger.info("=" * 70)
    logger.info("INITIALIZING SOCIAL LISTENING & TREND PREDICTION PIPELINE")
    logger.info(f"Project Name: {config.get('project', {}).get('name')}")
    logger.info("=" * 70)

    if args.stage is None and not args.all:
        logger.warning("Please specify --stage <1-8> or --all to execute. Use --help for usage details.")
        sys.exit(0)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Stage 1: Data Collection
    # --------------------------------------------------------------------------
    if args.all or args.stage == 1:
        logger.info("\n>>> [STAGE 1] DATA COLLECTION & INGESTION")
        from src.stage_01_data_collection import DataCollector
        collector = DataCollector(
            output_path=f"{paths.get('raw_data_dir', 'data/01_raw')}/raw_posts.parquet"
        )
        import os
        default_inputs = [
            "data/01_raw/input_data.parquet",
            "data/01_raw/social_listening_30days.csv",
            "data/01_raw/social_listening_dataset.csv",
            "data/01_raw/input_data.csv"
        ]
        input_file = args.input or next((p for p in default_inputs if os.path.exists(p)), "data/01_raw/input_data.parquet")
        try:
            collector.collect_from_file(input_file)
        except Exception as e:
            logger.warning(f"Stage 1 input file '{input_file}' not found: {e}. Please supply data to proceed.")
            if not args.all:
                return

    # --------------------------------------------------------------------------
    # Stage 2: Preprocessing Data
    # --------------------------------------------------------------------------
    if args.all or args.stage == 2:
        logger.info("\n>>> [STAGE 2] PREPROCESSING DATA")
        from src.stage_02_preprocessing import PreprocessingPipeline
        preprocessor = PreprocessingPipeline(config.get("stage_02_preprocessing", {}))
        input_path = f"{paths.get('raw_data_dir', 'data/01_raw')}/raw_posts.parquet"
        preprocessor.run_from_file(input_path)

    # --------------------------------------------------------------------------
    # Stage 3: Multilingual Representation & Sentiment Analysis (XLM-RoBERTa)
    # --------------------------------------------------------------------------
    if args.all or args.stage == 3:
        logger.info("\n>>> [STAGE 3] MULTILINGUAL EMBEDDINGS & SENTIMENT ANALYSIS")
        from src.stage_03_multilingual_sentiment import MultilingualSentimentPipeline
        nlp_pipeline = MultilingualSentimentPipeline(config.get("stage_03_multilingual_sentiment", {}))
        input_path = f"{paths.get('preprocessed_data_dir', 'data/02_preprocessed')}/preprocessed_posts.parquet"
        nlp_pipeline.run_from_file(input_path)

    # --------------------------------------------------------------------------
    # Stage 4: Topic Detection (BERTopic)
    # --------------------------------------------------------------------------
    if args.all or args.stage == 4:
        logger.info("\n>>> [STAGE 4] TOPIC DETECTION (BERTopic)")
        from src.stage_04_topic_detection import TopicDetector
        from src.utils.file_io import load_dataframe
        import numpy as np
        
        topic_detector = TopicDetector(config.get("stage_04_topic_detection", {}))
        posts_df = load_dataframe(f"{paths.get('embeddings_dir', 'data/03_embeddings_sentiment')}/posts_with_sentiment.parquet")
        embeds = np.load(f"{paths.get('embeddings_dir', 'data/03_embeddings_sentiment')}/sentence_embeddings.npy")
        topic_detector.fit_transform(posts_df, embeds)

    # --------------------------------------------------------------------------
    # Stage 5: Topic Tracking
    # --------------------------------------------------------------------------
    if args.all or args.stage == 5:
        logger.info("\n>>> [STAGE 5] TOPIC TRACKING OVER TIME")
        from src.stage_05_topic_tracking import TopicTracker
        tracker = TopicTracker(config.get("stage_05_topic_tracking", {}))
        input_path = f"{paths.get('topics_dir', 'data/04_topics')}/posts_with_topics.parquet"
        tracker.run_from_file(input_path)

    # --------------------------------------------------------------------------
    # Stage 6: Feature Engineering & Label Data
    # --------------------------------------------------------------------------
    if args.all or args.stage == 6:
        logger.info("\n>>> [STAGE 6] FEATURE ENGINEERING & LABELING")
        from src.stage_06_feature_engineering_labeling import FeatureAndLabelBuilder
        builder = FeatureAndLabelBuilder(config.get("stage_06_feature_engineering_labeling", {}))
        input_path = f"{paths.get('topic_tracking_dir', 'data/05_topic_tracking')}/tracked_topic_metrics.parquet"
        builder.run_from_file(input_path)

    # --------------------------------------------------------------------------
    # Stage 7: Time Based Data Splitting
    # --------------------------------------------------------------------------
    if args.all or args.stage == 7:
        logger.info("\n>>> [STAGE 7] TIME BASED DATA SPLITTING (70% - 20% - 10%)")
        from src.stage_07_time_based_split import TimeBasedSplitter
        splitter = TimeBasedSplitter(config.get("stage_07_time_based_split", {}))
        input_path = f"{paths.get('features_labels_dir', 'data/06_features_labels')}/features_and_labels.parquet"
        splitter.split_from_file(input_path)

    # --------------------------------------------------------------------------
    # Stage 8: Prediction Models & Feature Fusion
    # --------------------------------------------------------------------------
    if args.all or args.stage == 8:
        logger.info("\n>>> [STAGE 8] PREDICTION MODELS (SARIMA + FUSION + LIGHTGBM)")
        from src.stage_08_prediction_models import PredictionPipeline
        predictor = PredictionPipeline(config.get("stage_08_prediction_models", {}))
        splits_dir = paths.get("splits_dir", "data/07_splits")
        predictor.run_from_splits(splits_dir)

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETED ALL REQUESTED STAGES SUCCESSFULLY!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
