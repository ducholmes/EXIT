#!/usr/bin/env python3
"""Gemini API reader with a persistent, conservative free-tier quota guard."""

from __future__ import annotations

import calendar
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Deque, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


try:
    PACIFIC = ZoneInfo("America/Los_Angeles")
except ZoneInfoNotFoundError:  # Windows Python may not have an IANA database yet.
    PACIFIC = None
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


def pacific_now() -> datetime:
    """Return Pacific wall time, with a US DST fallback if tzdata is absent."""

    current_utc = datetime.now(timezone.utc)
    if PACIFIC is not None:
        return current_utc.astimezone(PACIFIC)

    year = current_utc.year

    def nth_sunday(month: int, occurrence: int) -> int:
        first_weekday, _ = calendar.monthrange(year, month)
        first_sunday = 1 + (6 - first_weekday) % 7
        return first_sunday + 7 * (occurrence - 1)

    daylight_starts = datetime(
        year, 3, nth_sunday(3, 2), 10, tzinfo=timezone.utc
    )
    daylight_ends = datetime(
        year, 11, nth_sunday(11, 1), 9, tzinfo=timezone.utc
    )
    is_daylight = daylight_starts <= current_utc < daylight_ends
    offset = timedelta(hours=-7 if is_daylight else -8)
    label = "PDT" if is_daylight else "PST"
    return current_utc.astimezone(timezone(offset, label))


class GeminiAPIError(RuntimeError):
    """An API request failed; quota errors are never retried automatically."""

    def __init__(self, status: Optional[int], message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"Gemini API HTTP {status}: {message}" if status else message)


class DailyQuotaReached(RuntimeError):
    """The safe daily request budget is exhausted for the Pacific calendar day."""


@dataclass(frozen=True)
class GeminiQuota:
    rpm: int
    input_tpm: int
    rpd: int
    safety_fraction: float = 0.8

    @classmethod
    def from_mapping(cls, values: Dict) -> "GeminiQuota":
        missing = [
            key for key in ("rpm", "input_tpm", "rpd")
            if key not in values or values[key] is None
        ]
        if missing:
            raise ValueError(
                "Gemini quota config is missing " + ", ".join(missing)
            )
        quota = cls(
            rpm=int(values["rpm"]),
            input_tpm=int(values["input_tpm"]),
            rpd=int(values["rpd"]),
            safety_fraction=float(values.get("safety_fraction", 0.8)),
        )
        if min(quota.rpm, quota.input_tpm, quota.rpd) < 1:
            raise ValueError("Gemini RPM, input TPM, and RPD must all be positive")
        if not 0.0 < quota.safety_fraction <= 1.0:
            raise ValueError("safety_fraction must be greater than 0 and at most 1")
        if min(quota.safe_rpm, quota.safe_input_tpm, quota.safe_rpd) < 1:
            raise ValueError(
                "The configured quota multiplied by safety_fraction rounds to zero; "
                "raise the quota or safety_fraction."
            )
        return quota

    @property
    def safe_rpm(self) -> int:
        return math.floor(self.rpm * self.safety_fraction)

    @property
    def safe_input_tpm(self) -> int:
        return math.floor(self.input_tpm * self.safety_fraction)

    @property
    def safe_rpd(self) -> int:
        return math.floor(self.rpd * self.safety_fraction)


class QuotaGuard:
    """Track this runner's usage and stay below all configured dimensions."""

    def __init__(
        self, model: str, project_id: str, quota: GeminiQuota, state_path: Path
    ) -> None:
        self.model = model
        self.project_id = project_id
        self.quota = quota
        self.state_path = state_path
        self.events: Deque[Tuple[float, int]] = deque()
        self.pacific_date = ""
        self.daily_requests = 0
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read quota state {self.state_path}: {exc}") from exc
        if state.get("model") != self.model or state.get("project_id") != self.project_id:
            raise ValueError(
                "Quota state belongs to a different Gemini model/project; "
                "use a separate state file."
            )
        now = time.time()
        self.events = deque(
            (float(event["time"]), int(event["input_tokens"]))
            for event in state.get("events", [])
            if now - float(event["time"]) < 60.0
        )
        current_date = pacific_now().date().isoformat()
        if state.get("pacific_date") == current_date:
            self.pacific_date = current_date
            self.daily_requests = int(state.get("daily_requests", 0))

    def _sync_date(self) -> None:
        current_date = pacific_now().date().isoformat()
        if self.pacific_date != current_date:
            self.pacific_date = current_date
            self.daily_requests = 0

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "project_id": self.project_id,
                    "pacific_date": self.pacific_date,
                    "daily_requests": self.daily_requests,
                    "events": [
                        {"time": timestamp, "input_tokens": tokens}
                        for timestamp, tokens in self.events
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.state_path)

    def reserve_generation(self, input_tokens: int) -> None:
        if input_tokens < 0:
            raise ValueError("input token count cannot be negative")
        if input_tokens > self.quota.safe_input_tpm:
            raise ValueError(
                f"One prompt has {input_tokens} input tokens, exceeding the safe "
                f"TPM budget of {self.quota.safe_input_tpm}; no generation request sent."
            )

        while True:
            now = time.time()
            self._sync_date()
            while self.events and now - self.events[0][0] >= 60.0:
                self.events.popleft()

            if self.daily_requests >= self.quota.safe_rpd:
                next_reset = pacific_now()
                next_reset = (next_reset + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                raise DailyQuotaReached(
                    f"Safe RPD budget ({self.quota.safe_rpd}) is used for "
                    f"{self.pacific_date}; resume after {next_reset.isoformat()}."
                )

            used_tokens = sum(tokens for _, tokens in self.events)
            rpm_full = len(self.events) >= self.quota.safe_rpm
            tpm_full = used_tokens + input_tokens > self.quota.safe_input_tpm
            if not rpm_full and not tpm_full:
                # Reserve before sending so a crash or API failure cannot cause
                # this runner to exceed its configured daily budget on resume.
                self.events.append((now, input_tokens))
                self.daily_requests += 1
                self._save()
                return

            wait_until = self.events[0][0] + 60.0 if self.events else now + 60.0
            if tpm_full:
                running_tokens = used_tokens
                for timestamp, tokens in self.events:
                    running_tokens -= tokens
                    if running_tokens + input_tokens <= self.quota.safe_input_tpm:
                        wait_until = max(wait_until, timestamp + 60.0)
                        break
            wait_seconds = max(0.1, wait_until - now + 0.05)
            reason = "RPM" if rpm_full else "input TPM"
            print(f"Gemini safe {reason} budget reached; waiting {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)


class GeminiAPIReader:
    """Gemma reader hosted by the Gemini API."""

    def __init__(
        self,
        model: str,
        project_id: str,
        quota: GeminiQuota,
        quota_state_path: Path,
        api_key: Optional[str] = None,
        max_new_tokens: int = 100,
        timeout_seconds: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY in the environment before using Gemini")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.guard = QuotaGuard(model, project_id, quota, quota_state_path)
        self.last_input_tokens = 0
        self.last_usage: Dict[str, int] = {}

    def _post(self, method: str, payload: Dict) -> Dict:
        url = f"{API_ROOT}/{quote(self.model, safe='')}:{method}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("error", {}).get("message", body)
            except json.JSONDecodeError:
                detail = body
            raise GeminiAPIError(exc.code, str(detail)) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise GeminiAPIError(None, str(exc)) from exc

    @staticmethod
    def _contents(prompt: str) -> Dict:
        return {"role": "user", "parts": [{"text": prompt}]}

    def count_input_tokens(self, prompt: str) -> int:
        response = self._post(
            "countTokens",
            {"contents": [self._contents(prompt)]},
        )
        token_count = response.get("totalTokens")
        if not isinstance(token_count, int):
            raise GeminiAPIError(None, "countTokens response omitted totalTokens")
        return token_count

    def generate(self, prompt: str) -> Tuple[str, float]:
        input_tokens = self.count_input_tokens(prompt)
        self.guard.reserve_generation(input_tokens)
        generation_config = {"maxOutputTokens": self.max_new_tokens}
        if self.model.startswith("gemma-4-"):
            generation_config["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}
        payload = {
            "contents": [self._contents(prompt)],
            "generationConfig": generation_config,
        }
        started_at = time.perf_counter()
        response = self._post("generateContent", payload)
        elapsed = time.perf_counter() - started_at
        candidates = response.get("candidates") or []
        if not candidates:
            raise GeminiAPIError(None, "generateContent response contained no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "".join(part.get("text", "") for part in parts).strip()
        usage = response.get("usageMetadata") or {}
        self.last_input_tokens = int(usage.get("promptTokenCount", input_tokens))
        self.last_usage = {
            "input_tokens": self.last_input_tokens,
            "output_tokens": int(usage.get("candidatesTokenCount", 0)),
            "thought_tokens": int(usage.get("thoughtsTokenCount", 0)),
        }
        return answer, elapsed
