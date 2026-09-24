from enum import Enum


class DocumentProcessingStatus(
    str,
    Enum,
):
    UPLOADED = "uploaded"

    QUEUED = "queued"

    PROCESSING = "processing"

    COMPLETED = "completed"

    FAILED = "failed"