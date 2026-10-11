import re
import pandas as pd
from loguru import logger
from typing import List, Optional

DEFAULT_GAMBLING_SPAM = [
    "kèo nhà cái", "tỷ lệ kèo", "kèo bóng", "soi kèo", "nhóm tele", "link cá cược",
    "nhà cái uy tín", "cược chấp", "tài xỉu", "đặt cược", "nhận định kèo", "kèo vip"
]

def filter_spam_and_bots(
    df: pd.DataFrame,
    min_followers: int = 5,
    max_hashtags: int = 15,
    max_urls: int = 4,
    spam_keywords: Optional[List[str]] = None,
    gambling_keywords: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Step 2: Remove Spam Bot & Sports Betting Scams
    Filters out automated bot accounts, promotional spam, and betting links based on:
    - Extremely low follower count (fake bot farms)
    - Excessive hashtag stuffing
    - Abnormal URL link density
    - Sports gambling / betting spam keyword patterns
    """
    initial_count = len(df)
    
    # 1. Filter by author follower count
    if "followers" in df.columns:
        mask_followers = df["followers"] >= min_followers
    else:
        mask_followers = pd.Series(True, index=df.index)
    
    # 2. Filter by hashtag count
    def count_hashtags(val):
        if isinstance(val, list):
            return len(val)
        if isinstance(val, str):
            return len(re.findall(r"#\w+", val))
        return 0
    
    if "hashtags" in df.columns:
        mask_hashtags = df["hashtags"].apply(count_hashtags) <= max_hashtags
    else:
        mask_hashtags = pd.Series(True, index=df.index)
    
    # 3. Filter by URL count
    def count_urls(text):
        if not isinstance(text, str):
            return 0
        return len(re.findall(r"https?://\S+|www\.\S+", text))
    
    mask_urls = df["text"].apply(count_urls) <= max_urls
    
    # 4. Filter by spam & sports betting keywords
    all_spam_words = list(spam_keywords or []) + list(gambling_keywords or DEFAULT_GAMBLING_SPAM)
    if all_spam_words:
        pattern = "|".join([re.escape(k) for k in all_spam_words if k])
        mask_keywords = ~df["text"].str.contains(pattern, case=False, na=False)
    else:
        mask_keywords = pd.Series(True, index=df.index)
        
    final_mask = mask_followers & mask_hashtags & mask_urls & mask_keywords
    df_clean = df[final_mask].reset_index(drop=True)
    
    removed_count = initial_count - len(df_clean)
    logger.info(f"[Remove Spam Bot & Betting] Filtered out {removed_count} spam/gambling posts. Remaining: {len(df_clean)} posts.")
    return df_clean
