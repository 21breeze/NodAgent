from typing import List

from sqlalchemy.orm import Session

from backend.app.models.workspace import Workspace
from backend.app.schemas.workspace import WorkspaceCreate


def get_workspaces(db: Session) -> List[Workspace]:
    return (
        db.query(Workspace)
        .order_by(Workspace.id)
        .all()
    )


def create_workspace(
    db: Session,
    workspace: WorkspaceCreate,
) -> Workspace:
    db_workspace = Workspace(
        name=workspace.name,
        description=workspace.description,
    )

    db.add(db_workspace)

    db.commit()

    db.refresh(db_workspace)

    return db_workspace