import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyst.agent import DataAnalystAgent  # noqa: E402
from analyst.data_gen import generate  # noqa: E402
from analyst.db import Database  # noqa: E402


@pytest.fixture(scope="session")
def frames():
    return generate()


@pytest.fixture(scope="session")
def db(frames):
    return Database.from_frames(frames[0])


@pytest.fixture(scope="session")
def agent(db):
    return DataAnalystAgent(db, None)


@pytest.fixture(scope="session")
def q3(agent):
    return agent.investigate("Why did revenue drop in Q3?")


class FakeLLM:
    """Scripted model: returns queued replies; records prompts."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, system, user, max_tokens=1500):
        self.calls.append((system, user))
        return self.replies.pop(0)
