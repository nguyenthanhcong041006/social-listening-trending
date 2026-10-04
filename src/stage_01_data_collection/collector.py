import pandas as pd
from typing import List, Dict, Any
from loguru import logger
from .schema import RawPostSchema
from src.utils.file_io import save_dataframe, load_dataframe, resolve_path, resolve_output_path

# Standard alias map for common social media column variations
COLUMN_ALIASES = {
    "text": ["text", "text_content", "content", "tweet", "caption", "message", "body", "post_text"],
    "timestamp": ["timestamp", "created_at", "date", "datetime", "post_time", "time"],
    "likes": ["likes", "likes_count", "like_count", "favorite_count", "favs"],
    "shares": ["shares", "shares_count", "share_count", "retweet_count", "retweets", "reposts"],
    "comments": ["comments", "comments_count", "comment_count", "replies", "reply_count"],
    "hashtags": ["hashtags", "hashtag_list", "tags"],
    "followers": ["followers", "followers_count", "follower_count", "user_followers", "author_followers", "userfollowers"],
    "post_id": ["post_id", "id", "tweet_id", "uid", "post_uid", "url"],
    "platform": ["platform", "source", "network"]
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically maps common variations of column names to the standard schema:
    text, timestamp, likes, shares, comments, hashtags, followers.
    """
    import re
    df = df.copy()
    col_mapping = {}
    
    # Lowercase all existing columns for case-insensitive matching
    existing_cols = {col.lower().strip(): col for col in df.columns}
    
    for standard_name, aliases in COLUMN_ALIASES.items():
        if standard_name in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in existing_cols:
                original_col = existing_cols[alias.lower()]
                col_mapping[original_col] = standard_name
                break
                
    if col_mapping:
        logger.info(f"Auto-mapped dataset columns: {col_mapping}")
        df = df.rename(columns=col_mapping)
        
    # Safe numeric casting for engagement metrics
    for num_col in ["likes", "shares", "comments", "followers"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0).astype(int).clip(lower=0)

    if "post_id" in df.columns:
        df["post_id"] = df["post_id"].astype(str)

    # If comments is missing, default to 0
    if "comments" not in df.columns:
        df["comments"] = 0

    # If hashtags is missing, extract from text
    if "hashtags" not in df.columns and "text" in df.columns:
        logger.info("Extracting hashtags automatically from text using regex #(\\w+)...")
        df["hashtags"] = df["text"].astype(str).apply(lambda t: re.findall(r"#(\w+)", t))

    # If followers is missing, provide a safe fallback (e.g. from impressions or default)
    if "followers" not in df.columns:
        if "impressions" in df.columns:
            logger.info("Column 'followers' missing: Estimated proxy from 'impressions // 10'.")
            df["followers"] = (df["impressions"] // 10).clip(lower=1)
        else:
            logger.info("Column 'followers' missing: Defaulting to baseline value (100).")
            df["followers"] = 100

    return df

class DataCollector:
    """
    Ingests, standardizes, and validates raw social media data (Stage 1: Data Collection).
    Validates mandatory schema fields: Text, Timestamp, Likes, Shares, Comments, Hashtags, Followers.
    """

    def __init__(self, output_path: str = "data/01_raw/raw_social_posts.parquet"):
        self.output_path = resolve_output_path(output_path)

    def validate_and_ingest(self, raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Validates raw input dictionaries against Pydantic RawPostSchema.
        """
        valid_posts = []
        errors_count = 0
        
        for record in raw_records:
            try:
                post = RawPostSchema(**record)
                valid_posts.append(post.model_dump())
            except Exception as e:
                errors_count += 1
                logger.debug(f"Invalid record dropped: {e}")

        logger.info(f"Successfully validated {len(valid_posts)} posts. Dropped {errors_count} malformed posts.")
        df = pd.DataFrame(valid_posts)
        return df

    def collect_from_file(self, input_file_path: str) -> pd.DataFrame:
        """Loads data from a source file (CSV/JSON), standardizes columns, and validates schema."""
        resolved_path = resolve_path(input_file_path)
        logger.info(f"Collecting data from input file: {resolved_path}")
        raw_df = load_dataframe(resolved_path)
        
        # Standardize column variations
        standardized_df = standardize_columns(raw_df)
        records = standardized_df.to_dict(orient="records")
        clean_df = self.validate_and_ingest(records)
        
        save_dataframe(clean_df, self.output_path)
        return clean_df
