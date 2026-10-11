import re
import pandas as pd
from typing import List, Dict, Any
from loguru import logger
from .schema import RawPostSchema
from src.utils.file_io import save_dataframe, load_dataframe, resolve_path, resolve_output_path

# Standard alias map for common social media column variations across platforms
COLUMN_ALIASES = {
    "text": ["text", "text_content", "content", "tweet", "caption", "message", "body", "post_text", "title_and_text"],
    "timestamp": ["timestamp", "created_at", "date", "datetime", "post_time", "time", "published_at"],
    "likes": ["likes", "likes_count", "like_count", "favorite_count", "favs", "upvotes", "reactions"],
    "shares": ["shares", "shares_count", "share_count", "retweet_count", "retweets", "reposts"],
    "comments": ["comments", "comments_count", "comment_count", "replies", "reply_count", "num_comments"],
    "views": ["views", "view_count", "play_count", "video_views", "plays", "impressions"],
    "hashtags": ["hashtags", "hashtag_list", "tags"],
    "followers": ["followers", "followers_count", "follower_count", "user_followers", "author_followers", "subscribers"],
    "post_id": ["post_id", "id", "tweet_id", "uid", "post_uid", "url", "video_id"],
    "platform": ["platform", "source", "network", "channel", "site", "source_platform"],
    "sport_category": ["sport_category", "sport", "category", "mon_the_thao", "league"],
    "source_type": ["source_type", "channel_type", "type", "page_type"]
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes raw multi-platform social media posts into a uniform schema:
    text, timestamp, likes, shares, comments, views, hashtags, followers, platform, sport_category.
    """
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
    for num_col in ["likes", "shares", "comments", "views", "followers"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0).astype(int).clip(lower=0)
        else:
            df[num_col] = 0

    if "post_id" in df.columns:
        df["post_id"] = df["post_id"].astype(str)

    # Standardize platform names
    if "platform" not in df.columns:
        df["platform"] = "generic"
    else:
        alias_map = {
            "fb": "facebook",
            "yt": "youtube",
            "ytb": "youtube",
            "tt": "tiktok",
            "tik_tok": "tiktok",
            "twitter": "x",
            "reddit": "reddit",
            "news": "news",
            "bao": "news"
        }
        df["platform"] = df["platform"].astype(str).str.lower().str.strip().replace(alias_map)

    # Sport category default
    if "sport_category" not in df.columns:
        df["sport_category"] = "sports"

    # If hashtags is missing, extract from text
    if "hashtags" not in df.columns and "text" in df.columns:
        logger.info("Extracting hashtags automatically from text using regex #(\w+)...")
        df["hashtags"] = df["text"].astype(str).apply(lambda t: re.findall(r"#(\w+)", t))

    # If followers is missing, provide a safe fallback from views/impressions or default
    if df["followers"].sum() == 0:
        if df["views"].sum() > 0:
            logger.info("Column 'followers' missing: Estimated proxy from 'views // 10'.")
            df["followers"] = (df["views"] // 10).clip(lower=1)
        else:
            df["followers"] = 100

    return df

class DataCollector:
    """
    Ingests, standardizes, and validates raw multi-platform sports social data (Stage 1).
    Supports Facebook, TikTok, YouTube, Reddit, News with uniform schema.
    """

    def __init__(self, output_path: str = "data/01_raw/raw_posts.parquet"):
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

    def collect_from_file(self, input_file: str) -> pd.DataFrame:
        """
        Loads raw data from file (CSV, Parquet, JSON), standardizes columns, validates, and saves.
        """
        resolved_input = resolve_path(input_file)
        logger.info(f"Ingesting raw multi-platform sports social data from: {resolved_input}")
        
        df_raw = load_dataframe(resolved_input)
        logger.info(f"Raw data loaded: {len(df_raw)} records across {len(df_raw.columns)} columns.")
        
        df_standard = standardize_columns(df_raw)
        records = df_standard.to_dict(orient="records")
        df_valid = self.validate_and_ingest(records)
        
        save_dataframe(df_valid, self.output_path)
        logger.info(f"Saved standardized multi-platform sports data to: {self.output_path}")
        return df_valid
