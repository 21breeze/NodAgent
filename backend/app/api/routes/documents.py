import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from backend.app.core.document_status import (
    DocumentProcessingStatus,
)
from backend.app.db.database import (
    get_db,
)
from backend.app.models.document import (
    Document,
)
from backend.app.models.document_chunk import (
    DocumentChunk,
)
from backend.app.models.workspace import (
    Workspace,
)
from backend.app.schemas.document import (
    DocumentChunkResponse,
    DocumentProcessingResponse,
    DocumentResponse,
)
from backend.app.services import (
    document_action_service,
)
from backend.app.services.document_action_service import (
    DocumentBusyError,
    DocumentNotFoundError,
)
from backend.app.services.task_dispatcher import (
    DocumentAlreadyProcessingError,
    TaskDispatchError,
    dispatch_document_processing,
)


router = APIRouter(
    prefix=(
        "/workspaces/"
        "{workspace_id}/documents"
    ),
    tags=[
        "documents",
    ],
)


UPLOAD_DIR = Path(
    "data/uploads"
)


ALLOWED_SUFFIXES = {
    ".pdf",
    ".txt",
    ".md",
}


def get_workspace_or_404(
    db: Session,
    workspace_id: int,
) -> Workspace:
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.id
            == workspace_id
        )
        .first()
    )

    if workspace is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Workspace not found",
        )

    return workspace


def get_document_or_404(
    db: Session,
    workspace_id: int,
    document_id: int,
) -> Document:
    document = (
        db.query(Document)
        .filter(
            Document.id
            == document_id,
            Document.workspace_id
            == workspace_id,
        )
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Document not found",
        )

    return document


@router.get(
    "",
    response_model=list[
        DocumentResponse
    ],
)
def list_documents(
    workspace_id: int,
    db: Session = Depends(
        get_db
    ),
):
    get_workspace_or_404(
        db=db,
        workspace_id=workspace_id,
    )

    return (
        db.query(Document)
        .filter(
            Document.workspace_id
            == workspace_id
        )
        .order_by(
            Document.id.desc()
        )
        .all()
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=(
        status.HTTP_202_ACCEPTED
    ),
)
def upload_document(
    workspace_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(
        get_db
    ),
):
    get_workspace_or_404(
        db=db,
        workspace_id=workspace_id,
    )

    original_filename = (
        file.filename
        or "document"
    )

    suffix = (
        Path(
            original_filename
        )
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Unsupported file type. "
                "Supported types: "
                "pdf, txt, md"
            ),
        )

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stored_filename = (
        f"{uuid4().hex}"
        f"{suffix}"
    )

    file_path = (
        UPLOAD_DIR
        / stored_filename
    )

    try:
        with file_path.open(
            "wb"
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

    except Exception as exc:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Failed to save file",
        ) from exc

    if (
        not file_path.exists()
        or file_path.stat().st_size == 0
    ):
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="Uploaded file is empty",
        )

    document = Document(
        workspace_id=workspace_id,
        filename=original_filename,
        file_path=str(
            file_path
        ),
        content_type=(
            file.content_type
        ),
        processing_status=(
            DocumentProcessingStatus
            .UPLOADED
            .value
        ),
    )

    try:
        db.add(document)
        db.commit()
        db.refresh(document)

    except Exception as exc:
        db.rollback()

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to create "
                "document"
            ),
        ) from exc

    try:
        dispatch_document_processing(
            db=db,
            document=document,
        )

    except TaskDispatchError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Document was uploaded, "
                "but processing could not "
                "be queued"
            ),
        ) from exc

    db.refresh(document)

    return document


@router.post(
    "/{document_id}/process",
    response_model=(
        DocumentProcessingResponse
    ),
    status_code=(
        status.HTTP_202_ACCEPTED
    ),
)
def process_document_again(
    workspace_id: int,
    document_id: int,
    db: Session = Depends(
        get_db
    ),
):
    document = (
        get_document_or_404(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
        )
    )

    try:
        task_id = (
            dispatch_document_processing(
                db=db,
                document=document,
            )
        )

    except (
        DocumentAlreadyProcessingError
    ) as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=str(exc),
        ) from exc

    except TaskDispatchError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Failed to queue "
                "document processing"
            ),
        ) from exc

    db.refresh(document)

    return DocumentProcessingResponse(
        document_id=document.id,
        status=(
            document.processing_status
        ),
        task_id=task_id,
        error=(
            document.processing_error
        ),
    )


@router.get(
    "/{document_id}/status",
    response_model=(
        DocumentProcessingResponse
    ),
)
def get_document_status(
    workspace_id: int,
    document_id: int,
    db: Session = Depends(
        get_db
    ),
):
    document = (
        get_document_or_404(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
        )
    )

    return DocumentProcessingResponse(
        document_id=document.id,
        status=(
            document.processing_status
        ),
        task_id=(
            document.processing_task_id
        ),
        error=(
            document.processing_error
        ),
    )


@router.get(
    "/{document_id}/chunks",
    response_model=list[
        DocumentChunkResponse
    ],
)
def list_document_chunks(
    workspace_id: int,
    document_id: int,
    db: Session = Depends(
        get_db
    ),
):
    document = (
        get_document_or_404(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
        )
    )

    return (
        db.query(
            DocumentChunk
        )
        .filter(
            DocumentChunk.document_id
            == document.id
        )
        .order_by(
            DocumentChunk
            .chunk_index
            .asc()
        )
        .all()
    )


@router.delete(
    "/{document_id}",
    status_code=(
        status.HTTP_204_NO_CONTENT
    ),
)
def delete_document(
    workspace_id: int,
    document_id: int,
):
    try:
        document_action_service.delete_document(
            workspace_id=workspace_id,
            document_id=document_id,
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(exc),
        ) from exc

    except DocumentBusyError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to delete "
                "document"
            ),
        ) from exc

    return Response(
        status_code=(
            status.HTTP_204_NO_CONTENT
        )
    )
