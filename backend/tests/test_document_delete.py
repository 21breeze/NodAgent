import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db.database import Base
from backend.app.graphs.main_graph import (
    prepare_delete_node,
)
from backend.app.models.document import Document
from backend.app.models.workspace import Workspace
from backend.app.services import (
    document_action_service,
)


class DocumentDeleteTest(
    unittest.IsolatedAsyncioTestCase
):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={
                "check_same_thread": False
            },
            poolclass=StaticPool,
        )

        Base.metadata.create_all(
            self.engine,
            tables=[
                Workspace.__table__,
                Document.__table__,
            ],
        )

        self.session_factory = sessionmaker(
            bind=self.engine
        )

        with self.engine.begin() as connection:
            connection.execute(
                Workspace.__table__.insert(),
                [
                    {
                        "id": 1,
                        "name": "First",
                    },
                    {
                        "id": 2,
                        "name": "Second",
                    },
                ],
            )

            connection.execute(
                Document.__table__.insert(),
                [
                    {
                        "id": 1,
                        "workspace_id": 1,
                        "filename": "report.pdf",
                        "file_path": "report.pdf",
                        "processing_status": "completed",
                    },
                    {
                        "id": 2,
                        "workspace_id": 2,
                        "filename": "report.pdf",
                        "file_path": "other-report.pdf",
                        "processing_status": "completed",
                    },
                    {
                        "id": 3,
                        "workspace_id": 1,
                        "filename": "duplicate.pdf",
                        "file_path": "duplicate-1.pdf",
                        "processing_status": "completed",
                    },
                    {
                        "id": 4,
                        "workspace_id": 1,
                        "filename": "duplicate.pdf",
                        "file_path": "duplicate-2.pdf",
                        "processing_status": "completed",
                    },
                    {
                        "id": 5,
                        "workspace_id": 1,
                        "filename": "busy.pdf",
                        "file_path": "busy.pdf",
                        "processing_status": "processing",
                    },
                ],
            )

        self.session_patch = patch.object(
            document_action_service,
            "SessionLocal",
            self.session_factory,
        )

        self.session_patch.start()
        self.runtime = SimpleNamespace(
            context={
                "workspace_id": 1
            }
        )

    def tearDown(self):
        self.session_patch.stop()
        self.engine.dispose()

    async def test_exact_filename_resolves_in_workspace(self):
        result = await prepare_delete_node(
            {
                "action_document_id": None,
                "action_document_filename": (
                    "`report.pdf`"
                ),
            },
            self.runtime,
        )

        self.assertTrue(
            result["action_ready"]
        )
        self.assertEqual(
            result[
                "pending_action"
            ]["document_id"],
            1,
        )

    async def test_duplicate_filename_requires_id(self):
        result = await prepare_delete_node(
            {
                "action_document_id": None,
                "action_document_filename": (
                    "duplicate.pdf"
                ),
            },
            self.runtime,
        )

        self.assertFalse(
            result["action_ready"]
        )
        self.assertIn(
            "3, 4",
            result[
                "specialist_answer"
            ],
        )

    async def test_missing_filename_does_not_confirm(self):
        result = await prepare_delete_node(
            {
                "action_document_id": None,
                "action_document_filename": (
                    "missing.pdf"
                ),
            },
            self.runtime,
        )

        self.assertFalse(
            result["action_ready"]
        )
        self.assertIn(
            "没有找到",
            result[
                "specialist_answer"
            ],
        )

    async def test_processing_document_is_not_confirmed(self):
        result = await prepare_delete_node(
            {
                "action_document_id": None,
                "action_document_filename": (
                    "busy.pdf"
                ),
            },
            self.runtime,
        )

        self.assertFalse(
            result["action_ready"]
        )
        self.assertIn(
            "processing",
            result[
                "specialist_answer"
            ],
        )

    async def test_document_id_still_resolves(self):
        result = await prepare_delete_node(
            {
                "action_document_id": 1,
                "action_document_filename": None,
            },
            self.runtime,
        )

        self.assertTrue(
            result["action_ready"]
        )
        self.assertEqual(
            result[
                "pending_action"
            ]["document_id"],
            1,
        )

    async def test_mismatched_id_and_filename_fail(self):
        result = await prepare_delete_node(
            {
                "action_document_id": 1,
                "action_document_filename": (
                    "busy.pdf"
                ),
            },
            self.runtime,
        )

        self.assertFalse(
            result["action_ready"]
        )
        self.assertIn(
            "不一致",
            result[
                "specialist_answer"
            ],
        )


if __name__ == "__main__":
    unittest.main()
