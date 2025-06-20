"""
APIList Model Module

This module defines the APIList model for managing API keys and associated document data.
It provides functionality for storing and retrieving API keys linked to user documents
and their processing instructions.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime
from sqlalchemy.orm import relationship, Session
import os
from dotenv import load_dotenv
from models.__init__ import Base

load_dotenv()

class APIList(Base):
    """
    API List model for managing document-specific API keys.

    Attributes:
        id (int): Primary key
        main_table_user_id (int): Foreign key to users table
        api_key (str): Unique API key for document access
        document_id (int): Document identifier
        instructions (str): Processing instructions for the document
        created_at (datetime): Timestamp of API key creation
        last_used_at (datetime): Timestamp of last API key usage
        user (relationship): Relationship to User model
    """
    __tablename__ = 'api_list'

    id = Column(Integer, primary_key=True)
    label = Column(String, nullable=False)
    main_table_user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    api_key = Column(String(64), unique=True, nullable=False, index=True)
    instructions = Column(Text)
    created_at = Column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    last_used_at = Column(DateTime, nullable=True)
    total_tokens = Column(Integer, default=os.getenv("FREE_TOKENS", 50000))
    tokens_used = Column(Integer, default=0)
    tokens_remaining = Column(Integer, default=os.getenv("FREE_TOKENS", 50000))
    token_limit_per_day = Column(Integer)

    user = relationship("User", back_populates="api_keys")
    documents = relationship("Documents", back_populates="api", cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        if 'token_limit_per_day' in kwargs:
            token_limit = kwargs['token_limit_per_day']
            kwargs['tokens_remaining'] = token_limit
            kwargs['total_tokens'] = token_limit
        super().__init__(**kwargs)

    @classmethod
    def get_by_api_key(cls, db: Session, api_key: str):
        """
        Retrieve an API entry using its API key.

        Args:
            db: SQLAlchemy database session
            api_key: The API key string to search for

        Returns:
            APIList object if found, None otherwise
        """
        return db.query(cls).filter(cls.api_key == api_key).first()

    @classmethod
    def create_api_entry(cls, db: Session, main_table_user_id: int, api_key: str,
                         instructions: str = None, label:str = None, token_limit: int = None):
        """
        Create a new API key entry with associated document data.

        Args:
            db: SQLAlchemy database session
            main_table_user_id: User ID who owns this API key
            api_key: Unique API key string
            instructions: Optional processing instructions
            label: Optional label for the API key
            token_limit: Optional token limit per day

        Returns:
            Newly created APIList object
        """
        api_entry = cls(
            main_table_user_id=main_table_user_id,
            label=label,
            api_key=api_key,
            instructions=instructions,
            token_limit_per_day=token_limit or int(os.getenv("FREE_TOKENS", "1000"))
        )
        db.add(api_entry)
        db.commit()
        db.refresh(api_entry)
        return api_entry
