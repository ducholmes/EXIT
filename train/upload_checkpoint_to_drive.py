#!/usr/bin/env python3
"""Upload a local training checkpoint directory to Google Drive.

Authentication is provided by a Google service-account JSON file. Share the
destination Drive folder with the service account's ``client_email`` first.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from pathlib import Path
from typing import Dict, Optional


DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a checkpoint directory to a Google Drive folder"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local directory, e.g. /kaggle/working/exit-gemma-2b/final_model",
    )
    parser.add_argument(
        "--folder-id",
        required=True,
        help="ID of the destination Google Drive folder",
    )
    parser.add_argument(
        "--credentials",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Service-account JSON key path (or GOOGLE_APPLICATION_CREDENTIALS)",
    )
    parser.add_argument(
        "--name",
        help="Folder name to create in Drive (defaults to source directory name)",
    )
    return parser.parse_args()


def find_or_create_folder(service, name: str, parent_id: str) -> str:
    """Return the ID of a named child folder, creating it if necessary."""
    escaped_name = name.replace("\\", "\\\\").replace("'", "\\'")
    query = (
        f"'{parent_id}' in parents and name = '{escaped_name}' "
        "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    response = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id,name)",
        pageSize=100,
    ).execute()
    matches = response.get("files", [])
    if matches:
        return matches[0]["id"]

    created = service.files().create(
        body={
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        },
        fields="id",
    ).execute()
    return created["id"]


def upload_directory(service, source: Path, drive_parent_id: str) -> Dict[str, int]:
    """Recursively upload files, recreating subdirectories and replacing names."""
    folder_ids: Dict[str, str] = {".": drive_parent_id}
    uploaded = 0
    skipped = 0

    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink():
            print(f"Skipping symlink: {relative}")
            skipped += 1
            continue
        if path.is_dir():
            parent_relative = str(relative.parent)
            if parent_relative not in folder_ids:
                raise RuntimeError(f"Parent directory was not created: {relative}")
            folder_ids[str(relative)] = find_or_create_folder(
                service, path.name, folder_ids[parent_relative]
            )
            continue
        if not path.is_file():
            skipped += 1
            continue

        parent_relative = str(relative.parent)
        parent_id = folder_ids.get(parent_relative)
        if parent_id is None:
            raise RuntimeError(f"Parent directory was not created: {relative}")

        # Replace an existing file with the same name in the destination folder.
        escaped_name = path.name.replace("\\", "\\\\").replace("'", "\\'")
        response = service.files().list(
            q=f"'{parent_id}' in parents and name = '{escaped_name}' and trashed = false",
            spaces="drive",
            fields="files(id,name,mimeType)",
            pageSize=100,
        ).execute()
        existing = response.get("files", [])

        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(
            str(path),
            mimetype=mimetypes.guess_type(path.name)[0]
            or "application/octet-stream",
            resumable=True,
            chunksize=8 * 1024 * 1024,
        )
        if existing:
            # This is a training backup; if a name collides with a folder, do not
            # delete it or silently change its type.
            if existing[0].get("mimeType") == "application/vnd.google-apps.folder":
                raise RuntimeError(f"A Drive folder conflicts with file {relative}")
            service.files().update(
                fileId=existing[0]["id"], media_body=media
            ).execute()
            for duplicate in existing[1:]:
                service.files().delete(fileId=duplicate["id"]).execute()
        else:
            service.files().create(
                body={"name": path.name, "parents": [parent_id]},
                media_body=media,
                fields="id",
            ).execute()
        uploaded += 1
        print(f"Uploaded ({uploaded}): {relative}")

    return {"uploaded": uploaded, "skipped": skipped}


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        print(f"Source directory does not exist: {source}", file=sys.stderr)
        return 2
    if not args.folder_id:
        print("--folder-id is required", file=sys.stderr)
        return 2
    if not args.credentials:
        print(
            "Provide --credentials or set GOOGLE_APPLICATION_CREDENTIALS",
            file=sys.stderr,
        )
        return 2
    credentials_path = Path(args.credentials).expanduser()
    if not credentials_path.is_file():
        print(f"Credentials file does not exist: {credentials_path}", file=sys.stderr)
        return 2

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        print(
            "Install dependencies with: pip install google-api-python-client google-auth",
            file=sys.stderr,
        )
        return 2

    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path), scopes=[DRIVE_SCOPE]
    )
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    drive_folder_id = find_or_create_folder(
        service, args.name or source.name, args.folder_id
    )
    result = upload_directory(service, source, drive_folder_id)
    print(
        f"Done. Uploaded {result['uploaded']} file(s), "
        f"skipped {result['skipped']}. Drive folder ID: {drive_folder_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
