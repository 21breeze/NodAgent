from pathlib import Path
from typing import (
    Any,
    Dict,
)

from backend.app.core.document_status import (
    DocumentProcessingStatus,
)
from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.document import (
    Document,
)
from backend.app.models.document_chunk import (
    DocumentChunk,
)


class DocumentActionError(
    RuntimeError
):
    pass


class DocumentNotFoundError(
    DocumentActionError
):
    pass


class DocumentBusyError(
    DocumentActionError
):
    pass


def get_document_snapshot(
    workspace_id: int,
    document_id: int,
) -> Dict[str, Any]:
    with SessionLocal() as db:
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
            raise DocumentNotFoundError(
                "Document not found"
            )

        return {
            "id": document.id,
            "workspace_id": (
                document.workspace_id
            ),
            "filename": (
                document.filename
            ),
            "file_path": (
                document.file_path
            ),
            "processing_status": (
                document.processing_status
            ),
        }


def delete_document(
    workspace_id: int,
    document_id: int,
) -> Dict[str, Any]:
    with SessionLocal() as db:
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
            raise DocumentNotFoundError(
                "Document not found"
            )

        if document.processing_status in {
            (
                DocumentProcessingStatus
                .QUEUED
                .value
            ),
            (
                DocumentProcessingStatus
                .PROCESSING
                .value
            ),
        }:
            raise DocumentBusyError(
                "Document cannot be "
                "deleted while it is "
                "queued or processing"
            )

        filename = (
            document.filename
        )

        file_path = Path(
            document.file_path
        )

        (
            db.query(
                DocumentChunk
            )
            .filter(
                DocumentChunk.document_id
                == document.id
            )
            .delete(
                synchronize_session=False
            )
        )

        db.delete(
            document
        )

        db.commit()

    try:
        if file_path.exists():
            file_path.unlink()

    except OSError:
        pass

    return {
        "document_id": (
            document_id
        ),
        "filename": filename,
    }