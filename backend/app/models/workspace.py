from sqlalchemy import Column, Integer, String, Text

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