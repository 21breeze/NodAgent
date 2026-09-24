import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.db.database import Base
from backend.app.models.agent_memory import (
    AgentMemory,
)
from backend.app.models.workspace import (
    Workspace,
)
from backend.app.services import (
    memory_service,
)


class MemoryServiceTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:"
        )

        Base.metadata.create_all(
            self.engine,
            tables=[
                Workspace.__table__,
                AgentMemory.__table__,
            ],
        )

        self.db = Session(self.engine)

        self.db.add_all(
            [
                Workspace(
                    id=1,
                    name="First",
                ),
                Workspace(
                    id=2,
                    name="Second",
                ),
            ]
        )

        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_user_memory_is_shared_across_workspaces(self):
        memory = memory_service.upsert_memory(
            db=self.db,
            workspace_id=1,
            memory_scope="user",
            memory_key="food_preference",
            memory_value="喜欢吃炸鸡",
            user_id="user02",
        )

        other_workspace = (
            memory_service.get_user_memories(
                db=self.db,
                workspace_id=2,
                user_id="user02",
            )
        )

        self.assertEqual(
            [item.id for item in other_workspace],
            [memory.id],
        )

        updated = memory_service.upsert_memory(
            db=self.db,
            workspace_id=2,
            memory_scope="user",
            memory_key="food_preference",
            memory_value="喜欢吃面条",
            user_id="user02",
        )

        self.assertEqual(updated.id, memory.id)
        self.assertEqual(
            updated.memory_value,
            "喜欢吃面条",
        )
        self.assertEqual(
            memory_service.get_user_memories(
                db=self.db,
                workspace_id=1,
                user_id="another_user",
            ),
            [],
        )

        self.assertTrue(
            memory_service.delete_memory_by_key(
                db=self.db,
                workspace_id=2,
                memory_scope="user",
                memory_key="food_preference",
                user_id="user02",
            )
        )

        self.assertEqual(
            memory_service.get_user_memories(
                db=self.db,
                workspace_id=1,
                user_id="user02",
            ),
            [],
        )

    def test_workspace_memory_stays_isolated(self):
        memory_service.upsert_memory(
            db=self.db,
            workspace_id=1,
            memory_scope="workspace",
            memory_key="project_type",
            memory_value="Knowledge Base",
        )

        self.assertEqual(
            memory_service.get_workspace_memories(
                db=self.db,
                workspace_id=2,
            ),
            [],
        )

        self.assertFalse(
            memory_service.delete_memory_by_key(
                db=self.db,
                workspace_id=2,
                memory_scope="workspace",
                memory_key="project_type",
            )
        )

    def test_existing_duplicate_user_memories_are_hidden(self):
        self.db.add_all(
            [
                AgentMemory(
                    workspace_id=1,
                    memory_scope="user",
                    user_id="user02",
                    memory_key="food_preference",
                    memory_value="旧值",
                ),
                AgentMemory(
                    workspace_id=2,
                    memory_scope="user",
                    user_id="user02",
                    memory_key="food_preference",
                    memory_value="新值",
                ),
            ]
        )

        self.db.commit()

        memories = memory_service.get_user_memories(
            db=self.db,
            workspace_id=1,
            user_id="user02",
        )

        self.assertEqual(len(memories), 1)
        self.assertEqual(
            memories[0].memory_value,
            "新值",
        )

        self.assertTrue(
            memory_service.delete_memory_by_key(
                db=self.db,
                workspace_id=1,
                memory_scope="user",
                memory_key="food_preference",
                user_id="user02",
            )
        )

        self.assertEqual(
            memory_service.get_user_memories(
                db=self.db,
                workspace_id=2,
                user_id="user02",
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
