from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator

class RawPostSchema(BaseModel):
    """
    Validation schema for social media post data.
    Enforces the 7 required fields specified in the pipeline architecture:
    - text: Post text content
    - timestamp: Publication time (ISO string or datetime)
    - likes: Number of likes
    - shares: Number of shares / retweets
    - comments: Number of comments
    - hashtags: List of hashtags
    - followers: Follower count of the author account
    """
    post_id: Optional[str] = Field(default=None, description="Unique identifier of the post")
    text: str = Field(..., min_length=1, description="Text body of the post")
    timestamp: Union[datetime, str] = Field(..., description="Post publication timestamp")
    likes: int = Field(default=0, ge=0, description="Count of likes")
    shares: int = Field(default=0, ge=0, description="Count of shares / retweets")
    comments: int = Field(default=0, ge=0, description="Count of comments")
    hashtags: Union[List[str], str] = Field(default_factory=list, description="List of hashtags present in the post")
    followers: int = Field(default=0, ge=0, description="Follower count of the post author")
    platform: Optional[str] = Field(default="generic", description="Social media platform (X, Threads, etc.)")

    @field_validator("hashtags", mode="before")
    def parse_hashtags(cls, v):
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import ast
                try:
                    return ast.literal_eval(v)
                except Exception:
                    pass
            return [h.strip() for h in v.split(",") if h.strip()]
        return v or []

class SocialDataBatch(BaseModel):
    """Batch container for multiple social media posts."""
    posts: List[RawPostSchema]
