import pandas as pd
from loguru import logger
from .deduplicator import remove_duplicates
from .spam_filter import filter_spam_and_bots
from .text_cleaner import clean_text
from .text_normalizer import normalize_text
from .semantic_preserver import preserve_semantic_information
from src.utils.file_io import save_dataframe, load_dataframe

class PreprocessingPipeline:
    """
    Stage 2 Pipeline: Comprehensive Social Media Data Preprocessing
    Sequential execution:
    1. Remove Duplicate ->
    2. Remove Spam Bot ->
    3. Text Cleaning (HTML, whitespace, URLs) ->
    4. Text Normalization (Unicode, case, repeated chars) ->
    5. Preserve Semantic Information (Hashtags, Emojis, Negation)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.output_path = self.config.get("output_path", "data/02_preprocessed/preprocessed_posts.parquet")

    def process_text_sample(self, text: str) -> str:
        """Executes text transformation pipeline on an individual text string."""
        # Step 3 & 4: Text Cleaning & Remove URLs
        text = clean_text(text, remove_url=True)
        # Step 5: Text Normalization
        text = normalize_text(text, form="NFKC", lowercase=True, max_repeats=2)
        # Step 6: Preserve Semantic Information
        text = preserve_semantic_information(text, demojize=False)
        return text

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the complete preprocessing pipeline on a DataFrame."""
        logger.info(f"Starting data preprocessing on {len(df)} records...")
        
        # Step 1: Remove Duplicate
        df = remove_duplicates(df, subset=["text"])
        
        # Step 2: Remove Spam Bot
        df = filter_spam_and_bots(df)
        
        # Step 3, 4, 5, 6: Text Processing
        logger.info("Executing text cleaning, normalization, and semantic preservation...")
        df["cleaned_text"] = df["text"].apply(self.process_text_sample)
        
        # Drop samples where cleaned text is empty
        df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)
        
        save_dataframe(df, self.output_path)
        logger.info(f"Preprocessing completed. Saved clean dataset ({len(df)} rows) to: {self.output_path}")
        return df

    def run_from_file(self, input_path: str) -> pd.DataFrame:
        """Loads raw dataset from file and runs the preprocessing pipeline."""
        df = load_dataframe(input_path)
        return self.run(df)
