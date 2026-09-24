from uuid import uuid4

from sqlalchemy.orm import Session

from backend.app.core.document_status import (
    DocumentProcessingStatus,
)
from backend.app.models.document import (
    Document,
)
from backend.app.tasks.document_tasks import (
    process_document,
)


class TaskDispatchError(
    RuntimeError
):
    pass


class DocumentAlreadyProcessingError(
    RuntimeError
):
    pass


def dispatch_document_processing(
    db: Session,
    document: Document,
) -> str:
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
        raise (
            DocumentAlreadyProcessingError(
                "Document is already "
                "being processed"
            )
        )

    task_id = str(
        uuid4()
    )

    document.processing_status = (
        DocumentProcessingStatus
        .QUEUED
        .value
    )

    document.processing_task_id = (
        task_id
    )

    document.processing_error = None

    db.commit()
    db.refresh(document)

    try:
        process_document.apply_async(
            args=[
                document.id,
            ],
            task_id=task_id,
            queue="documents",
        )

    except Exception as exc:
        document.processing_status = (
            DocumentProcessingStatus
            .FAILED
            .value
        )

        document.processing_error = (
            (
                "Failed to dispatch "
                "Celery task: "
                f"{exc}"
            )[:2000]
        )

        db.commit()

        raise TaskDispatchError(
            "Failed to dispatch "
            "document processing task"
        ) from exc

    return task_id