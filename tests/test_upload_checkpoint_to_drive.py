"""Mocked tests for uploading checkpoint directories to Google Drive."""

import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from train.upload_checkpoint_to_drive import find_or_create_folder, upload_directory


class FakeRequest:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result


class FakeFiles:
    def __init__(self, list_results=None):
        self.list_results = list(list_results or [])
        self.list_calls = []
        self.create_calls = []
        self.update_calls = []
        self.delete_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        result = self.list_results.pop(0) if self.list_results else {"files": []}
        return FakeRequest(result)

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return FakeRequest({"id": f"created-{len(self.create_calls)}"})

    def update(self, **kwargs):
        self.update_calls.append(kwargs)
        return FakeRequest({"id": kwargs["fileId"]})

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)
        return FakeRequest({})


class FakeService:
    def __init__(self, list_results=None):
        self._files = FakeFiles(list_results)

    def files(self):
        return self._files


@pytest.fixture
def fake_media_upload(monkeypatch):
    """Install a fake MediaFileUpload so tests need no Google client package."""
    calls = []

    class FakeMediaFileUpload:
        def __init__(self, filename, **kwargs):
            self.filename = filename
            self.kwargs = kwargs
            calls.append(self)

    http_module = types.ModuleType("googleapiclient.http")
    http_module.MediaFileUpload = FakeMediaFileUpload
    api_module = types.ModuleType("googleapiclient")
    api_module.http = http_module
    monkeypatch.setitem(sys.modules, "googleapiclient", api_module)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http_module)
    return calls


def test_find_or_create_folder_reuses_existing_folder():
    service = FakeService([{"files": [{"id": "folder-existing", "name": "model"}]}])

    folder_id = find_or_create_folder(service, "model", "parent-1")

    assert folder_id == "folder-existing"
    assert service._files.create_calls == []
    assert "'parent-1' in parents" in service._files.list_calls[0]["q"]


def test_find_or_create_folder_creates_folder_and_escapes_name():
    service = FakeService([{"files": []}])

    folder_id = find_or_create_folder(service, "O'Brien\\model", "parent-1")

    assert folder_id == "created-1"
    query = service._files.list_calls[0]["q"]
    assert "O\\'Brien\\\\model" in query
    assert service._files.create_calls[0]["body"] == {
        "name": "O'Brien\\model",
        "mimeType": "application/vnd.google-apps.folder",
        "parents": ["parent-1"],
    }


def test_upload_directory_uploads_files_and_creates_nested_folders(
    fake_media_upload,
):
    with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
        source = Path(temp_dir) / "checkpoint"
        nested = source / "config"
        nested.mkdir(parents=True)
        (source / "weights.bin").write_bytes(b"weights")
        (nested / "settings.json").write_text("{}", encoding="utf-8")
        # Root file, folder lookup, nested file lookup have no existing match.
        service = FakeService([{"files": []}, {"files": []}, {"files": []}])

        result = upload_directory(service, source, "drive-root")

    assert result == {"uploaded": 2, "skipped": 0}
    assert len(fake_media_upload) == 2
    created = service._files.create_calls
    assert [call["body"]["name"] for call in created] == [
        "config",
        "settings.json",
        "weights.bin",
    ]
    assert created[0]["body"]["parents"] == ["drive-root"]
    assert created[1]["body"]["parents"] == ["created-1"]
    assert created[2]["body"]["parents"] == ["drive-root"]


def test_upload_directory_updates_first_duplicate_and_deletes_rest(
    fake_media_upload,
):
    with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
        source = Path(temp_dir) / "checkpoint"
        source.mkdir()
        (source / "weights.bin").write_bytes(b"new weights")
        service = FakeService(
            [
                {
                    "files": [
                        {"id": "old-1", "mimeType": "application/octet-stream"},
                        {"id": "old-2", "mimeType": "application/octet-stream"},
                    ]
                }
            ]
        )

        result = upload_directory(service, source, "drive-root")

    assert result == {"uploaded": 1, "skipped": 0}
    assert service._files.update_calls[0]["fileId"] == "old-1"
    assert service._files.delete_calls == [{"fileId": "old-2"}]


def test_upload_directory_rejects_file_name_conflicting_with_drive_folder(
    fake_media_upload,
):
    with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
        source = Path(temp_dir) / "checkpoint"
        source.mkdir()
        (source / "weights.bin").write_bytes(b"weights")
        service = FakeService(
            [{"files": [{"id": "folder-id", "mimeType": "application/vnd.google-apps.folder"}]}]
        )

        with pytest.raises(RuntimeError, match="Drive folder conflicts with file weights.bin"):
            upload_directory(service, source, "drive-root")

    assert service._files.update_calls == []
    assert service._files.create_calls == []


def test_upload_directory_skips_symlinks():
    with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
        temp_path = Path(temp_dir)
        source = temp_path / "checkpoint"
        source.mkdir()
        target = temp_path / "outside.bin"
        target.write_bytes(b"outside")
        link = source / "linked.bin"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("Creating symlinks is not permitted on this system")
        service = FakeService()

        result = upload_directory(service, source, "drive-root")

    assert result == {"uploaded": 0, "skipped": 1}
    assert service._files.list_calls == []
