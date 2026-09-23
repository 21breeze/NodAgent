from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Column,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.orm import relationship

from backend.app.core.config import EMBEDDING_DIMENSION
from backend.app.db.database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    workspace_id = Column(
        Integer,
        ForeignKey("workspaces.id"),
        nullable=False,
        index=True,
    )

    document_id = Column(
        Integer,
        ForeignKey("documents.id"),
        nullable=False,
        index=True,
    )

    chunk_index = Column(
        Integer,
        nullable=False,
    )

    content = Column(
        Text,
        nullable=False,
    )

    metadata_json = Column(
        JSON,
        nullable=False,
        default=dict,
    )

    embedding = Column(
        Vector(EMBEDDING_DIMENSION),
        nullable=False,
    )

    document = relationship(
        "Document",
        back_populates="chunks",
    )