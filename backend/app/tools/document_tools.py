from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from sqlalchemy import func

from backend.app.agents.context import AgentContext
from backend.app.db.database import SessionLocal
from backend.app.models.document import Document
from backend.app.models.document_chunk import DocumentChunk


DocumentStatus = Literal[
    "uploaded",
    "queued",
    "processing",
    "completed",
    "failed",
]


def build_document_payload(
    document: Document,
) -> Dict[str, Any]:
    return {
        "id": document.id,
        "filename": document.filename,
        "content_type": document.content_type,
        "processing_status": (
            document.processing_status
        ),
        "processing_task_id": (
            document.processing_task_id
        ),
        "processing_error": (
            document.processing_error
        ),
    }


@tool(
    "list_workspace_documents",
    response_format="content_and_artifact",
)
def list_workspace_documents(
    runtime: ToolRuntime[AgentContext],
    status: Optional[DocumentStatus] = None,
    filename_keyword: Optional[str] = None,
    limit: int = 20,
) -> Tuple[
    str,
    Dict[str, Any],
]:
    """
    List documents in the current workspace.

    Use this tool when the user asks which files or
    documents exist, which documents have completed
    processing, which documents failed, or wants to
    find a document by filename.

    status can filter documents by processing status.

    filename_keyword can filter documents whose
    filename contains the given text.
    """

    workspace_id = runtime.context[
        "workspace_id"
    ]

    safe_limit = max(
        1,
        min(limit, 50),
    )

    with SessionLocal() as db:
        query = (
            db.query(Document)
            .filter(
                Document.workspace_id
                == workspace_id
            )
        )

        if status is not None:
            query = query.filter(
                Document.processing_status
                == status
            )

        if filename_keyword:
            query = query.filter(
                Document.filename.ilike(
                    f"%{filename_keyword}%"
                )
            )

        documents = (
            query
            .order_by(
                Document.id.desc()
            )
            .limit(safe_limit)
            .all()
        )

        document_payloads = [
            build_document_payload(
                document
            )
            for document in documents
        ]

    if not document_payloads:
        return (
            (
                "No matching documents were "
                "found in the current workspace."
            ),
            {
                "documents": [],
            },
        )

    lines: List[str] = []

    for document in document_payloads:
        line = (
            f"document_id={document['id']}; "
            f"filename={document['filename']}; "
            f"status="
            f"{document['processing_status']}"
        )

        if document[
            "content_type"
        ]:
            line += (
                "; content_type="
                f"{document['content_type']}"
            )

        if (
            document[
                "processing_error"
            ]
        ):
            line += (
                "; error="
                f"{document['processing_error']}"
            )

        lines.append(
            line
        )

    return (
        "\n".join(lines),
        {
            "documents": (
                document_payloads
            ),
        },
    )


@tool(
    "get_document_details",
    response_format="content_and_artifact",
)
def get_document_details(
    document_id: int,
    runtime: ToolRuntime[
        AgentContext
    ],
) -> Tuple[
    str,
    Dict[str, Any],
]:
    """
    Get metadata and processing details for one
    document in the current workspace.

    Use this when the user refers to a specific
    document ID or needs its processing status,
    filename, error information, or chunk count.
    """

    workspace_id = runtime.context[
        "workspace_id"
    ]

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
            return (
                (
                    "The requested document "
                    "does not exist in the "
                    "current workspace."
                ),
                {
                    "documents": [],
                },
            )

        chunk_count = (
            db.query(
                func.count(
                    DocumentChunk.id
                )
            )
            .filter(
                DocumentChunk.document_id
                == document.id
            )
            .scalar()
        )

        payload = (
            build_document_payload(
                document
            )
        )

        payload[
            "chunk_count"
        ] = int(
            chunk_count or 0
        )

    lines = [
        (
            f"document_id="
            f"{payload['id']}"
        ),
        (
            f"filename="
            f"{payload['filename']}"
        ),
        (
            "processing_status="
            f"{payload['processing_status']}"
        ),
        (
            "chunk_count="
            f"{payload['chunk_count']}"
        ),
    ]

    if payload[
        "content_type"
    ]:
        lines.append(
            (
                "content_type="
                f"{payload['content_type']}"
            )
        )

    if payload[
        "processing_task_id"
    ]:
        lines.append(
            (
                "processing_task_id="
                f"{payload['processing_task_id']}"
            )
        )

    if payload[
        "processing_error"
    ]:
        lines.append(
            (
                "processing_error="
                f"{payload['processing_error']}"
            )
        )

    return (
        "\n".join(lines),
        {
            "documents": [
                payload
            ],
        },
    )