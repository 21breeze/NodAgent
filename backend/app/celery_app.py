from celery import Celery

from backend.app.core.config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
)


celery_app = Celery(
    "nodagent",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        (
            "backend.app.tasks."
            "document_tasks"
        ),
    ],
)


celery_app.conf.update(
    task_serializer="json",

    result_serializer="json",

    accept_content=[
        "json",
    ],

    task_track_started=True,

    broker_connection_retry_on_startup=True,

    result_expires=3600,

    enable_utc=True,

    worker_prefetch_multiplier=1,

    task_routes={
        (
            "backend.app.tasks."
            "document_tasks."
            "process_document"
        ): {
            "queue": "documents",
        },
    },
)