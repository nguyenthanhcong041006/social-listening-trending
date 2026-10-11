from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

class RawPostSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    post_id: Optional[Union[str, int]] = Field(default=None, description="Unique identifier of the post")
    text: str = Field(..., min_length=1, description="Text body of the post")
    timestamp: Union[datetime, str] = Field(..., description="Post publication timestamp")
    likes: int = Field(default=0, ge=0, description="Count of likes / reactions / upvotes")
    shares: int = Field(default=0, ge=0, description="Count of shares / retweets / reposts")
    comments: int = Field(default=0, ge=0, description="Count of comments / replies")
    views: int = Field(default=0, ge=0, description="Video views / impressions")
    hashtags: Union[List[str], str] = Field(default_factory=list, description="List of hashtags present in the post")
    followers: int = Field(default=0, ge=0, description="Follower count of the post author")
    platform: str = Field(default="generic", description="Social media platform (facebook, tiktok, youtube, reddit, news, etc.)")
    sport_category: Optional[str] = Field(default="sports", description="Sport domain e.g. football, tennis, etc.")
    source_type: Optional[str] = Field(default=None, description="Source entity type: fanpage, group, channel, subreddit, news")

    @field_validator("post_id", mode="before")
    def parse_post_id(cls, v):
        if v is not None:
            return str(v)
        return None

    @field_validator("platform", mode="before")
    def normalize_platform(cls, v):
        if not v:
            return "generic"
        val = str(v).lower().strip()
        alias_map = {
            "fb": "facebook",
            "yt": "youtube",
            "ytb": "youtube",
            "tt": "tiktok",
            "tik_tok": "tiktok",
            "twitter": "x",
            "reddit": "reddit",
            "news": "news",
            "bao": "news",
            "press": "news"
        }
        return alias_map.get(val, val)

    @field_validator("hashtags", mode="before")
    def parse_hashtags(cls, v):
        if v is None:
            return []
        if hasattr(v, "tolist"):
            return [str(x) for x in v.tolist() if x is not None]
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v if x is not None]
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import ast
                try:
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, (list, tuple)):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
            return [h.strip() for h in v.split(",") if h.strip()]
        return []

class SocialDataBatch(BaseModel):
    """Batch container for multiple social media posts."""
    posts: List[RawPostSchema]
