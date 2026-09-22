from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.app.db.database import Base


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    name = Column(
        String(100),
        nullable=False,
    )

    description = Column(
        Text,
        nullable=True,
    )

    documents = relationship(
        "Document",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )