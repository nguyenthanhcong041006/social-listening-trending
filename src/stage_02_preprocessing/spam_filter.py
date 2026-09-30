import re
import pandas as pd
from loguru import logger
from typing import List, Optional

def filter_spam_and_bots(
    df: pd.DataFrame,
    min_followers: int = 5,
    max_hashtags: int = 15,
    max_urls: int = 4,
    spam_keywords: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Step 2: Remove Spam Bot
    Filters out automated bot accounts and spam promotional content based on:
    - Extremely low follower count (fake bot farms)
    - Excessive hashtag stuffing
    - Abnormal URL link density
    - Spam/scam keyword matching
    """
    initial_count = len(df)
    
    # 1. Filter by author follower count
    mask_followers = df["followers"] >= min_followers
    
    # 2. Filter by hashtag count
    def count_hashtags(val):
        if isinstance(val, list):
            return len(val)
        if isinstance(val, str):
            return len(re.findall(r"#\w+", val))
        return 0
    
    mask_hashtags = df["hashtags"].apply(count_hashtags) <= max_hashtags
    
    # 3. Filter by URL count
    def count_urls(text):
        if not isinstance(text, str):
            return 0
        return len(re.findall(r"https?://\S+|www\.\S+", text))
    
    mask_urls = df["text"].apply(count_urls) <= max_urls
    
    # 4. Filter by spam keywords (if provided)
    if spam_keywords:
        pattern = "|".join([re.escape(k) for k in spam_keywords])
        mask_keywords = ~df["text"].str.contains(pattern, case=False, na=False)
    else:
        mask_keywords = pd.Series(True, index=df.index)
        
    final_mask = mask_followers & mask_hashtags & mask_urls & mask_keywords
    df_clean = df[final_mask].reset_index(drop=True)
    
    removed_count = initial_count - len(df_clean)
    logger.info(f"[Remove Spam Bot] Filtered out {removed_count} spam/bot posts. Remaining: {len(df_clean)} posts.")
    return df_clean
