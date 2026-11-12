import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def q(tmp_path, monkeypatch):
    """Fresh lib/worker (and api if importable) bound to a temp queue."""
    monkeypatch.setenv("LLOCKET_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("LLOCKET_WAKE_SOCKET", "")   # tests that want one set their own
    import lib
    importlib.reload(lib)
    import worker
    importlib.reload(worker)
    lib.ensure_queue_dirs()
    return lib, worker
