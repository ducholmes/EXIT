"""Verify visible long-running activity without waiting 30 real seconds."""
import io
import threading
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.progress import activity, copy_stream


def test_heartbeat_reports_activity_and_counters(capsys):
    reported = threading.Event()
    def details():
        reported.set()
        return '12 vectors'
    with activity('slow test operation', details, interval=0.01):
        assert reported.wait(2)
    output = capsys.readouterr().out
    assert 'START: slow test operation' in output
    assert 'WORKING: slow test operation' in output
    assert '12 vectors' in output
    assert 'DONE: slow test operation' in output


def test_stream_copy_and_failure_logging(capsys):
    target = io.BytesIO()
    copy_stream(io.BytesIO(b'passage data'), target, 'extract fixture')
    assert target.getvalue() == b'passage data'
    with pytest.raises(ValueError):
        with activity('failed operation'):
            raise ValueError('private error detail')
    output = capsys.readouterr().out
    assert 'FAILED: failed operation (ValueError)' in output
    assert 'private error detail' not in output
