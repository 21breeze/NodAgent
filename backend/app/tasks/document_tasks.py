from backend.app.celery_app import (
    celery_app,
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
from backend.app.services.document_processing_service import (
    prepare_document_chunks,
)


@celery_app.task(
    name=(
        "backend.app.tasks."
        "document_tasks."
        "process_document"
    ),
    bind=True,
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_document(
    self,
    document_id: int,
):
    file_path = None
    workspace_id = None

    with SessionLocal() as db:
        document = (
            db.query(Document)
            .filter(
                Document.id
                == document_id
            )
            .first()
        )

        if document is None:
            raise ValueError(
                "Document not found: "
                f"{document_id}"
            )

        document.processing_status = (
            DocumentProcessingStatus
            .PROCESSING
            .value
        )

        document.processing_task_id = (
            self.request.id
        )

        document.processing_error = None

        file_path = (
            document.file_path
        )

        workspace_id = (
            document.workspace_id
        )

        db.commit()

    try:
        prepared_chunks = (
            prepare_document_chunks(
                file_path=file_path
            )
        )

        with SessionLocal() as db:
            document = (
                db.query(Document)
                .filter(
                    Document.id
                    == document_id
                )
                .first()
            )

            if document is None:
                raise ValueError(
                    "Document not found: "
                    f"{document_id}"
                )

            (
                db.query(
                    DocumentChunk
                )
                .filter(
                    DocumentChunk.document_id
                    == document_id
                )
                .delete(
                    synchronize_session=False
                )
            )

            for chunk in (
                prepared_chunks
            ):
                db_chunk = (
                    DocumentChunk(
                        workspace_id=(
                            workspace_id
                        ),
                        document_id=(
                            document_id
                        ),
                        chunk_index=(
                            chunk[
                                "chunk_index"
                            ]
                        ),
                        content=(
                            chunk[
                                "content"
                            ]
                        ),
                        metadata_json=(
                            chunk[
                                "metadata_json"
                            ]
                        ),
                        embedding=(
                            chunk[
                                "embedding"
                            ]
                        ),
                    )
                )

                db.add(
                    db_chunk
                )

            document.processing_status = (
                DocumentProcessingStatus
                .COMPLETED
                .value
            )

            document.processing_error = (
                None
            )

            db.commit()

        return {
            "document_id": (
                document_id
            ),
            "status": (
                DocumentProcessingStatus
                .COMPLETED
                .value
            ),
            "chunk_count": len(
                prepared_chunks
            ),
        }

    except Exception as exc:
        with SessionLocal() as db:
            document = (
                db.query(Document)
                .filter(
                    Document.id
                    == document_id
                )
                .first()
            )

            if document is not None:
                (
                    document
                    .processing_status
                ) = (
                    DocumentProcessingStatus
                    .FAILED
                    .value
                )

                (
                    document
                    .processing_error
                ) = str(exc)[:2000]

                db.commit()

        raise