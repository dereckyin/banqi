"""Model league/version manager for self-play opponent diversity."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelEntry:
    """Metadata for one model version in league."""

    name: str
    path: str
    elo: float
    step: int
    is_best: bool


class LeagueManager:
    """Tracks model versions, best model, and opponent sampling."""

    def __init__(
        self,
        model_dir: str | Path = "models",
        max_versions: int = 8,
        manifest_name: str = "league.json",
        seed: Optional[int] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.max_versions = max_versions
        self.manifest_path = self.model_dir / manifest_name
        self._rng = random.Random(seed)
        self.entries: List[ModelEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.entries = [ModelEntry(**entry) for entry in payload.get("entries", [])]

    def _save(self) -> None:
        payload = {"entries": [asdict(entry) for entry in self.entries]}
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def has_best(self) -> bool:
        return any(entry.is_best for entry in self.entries)

    def get_best(self) -> Optional[ModelEntry]:
        for entry in self.entries:
            if entry.is_best:
                return entry
        return None

    def add_version_from_checkpoint(
        self,
        source_checkpoint: str | Path,
        elo: float,
        step: int,
        promote_best: bool = False,
    ) -> ModelEntry:
        version_idx = self._next_version_index()
        version_name = f"model_v{version_idx}"
        version_path = self.model_dir / f"{version_name}.pt"
        shutil.copy2(source_checkpoint, version_path)

        if promote_best:
            self._clear_best_flag()
            best_path = self.model_dir / "model_best.pt"
            shutil.copy2(source_checkpoint, best_path)
            entry = ModelEntry(name=version_name, path=str(version_path), elo=elo, step=step, is_best=True)
        else:
            entry = ModelEntry(name=version_name, path=str(version_path), elo=elo, step=step, is_best=False)
        self.entries.append(entry)
        self._trim_versions()
        self._save()
        return entry

    def promote_existing_entry(self, entry_name: str) -> None:
        target = next((entry for entry in self.entries if entry.name == entry_name), None)
        if target is None:
            raise ValueError(f"No model entry named {entry_name}")
        self._clear_best_flag()
        target.is_best = True
        shutil.copy2(target.path, self.model_dir / "model_best.pt")
        self._save()

    def sample_opponent(self, include_best: bool = True) -> Optional[ModelEntry]:
        if not self.entries:
            return None
        candidates = [entry for entry in self.entries if include_best or not entry.is_best]
        if not candidates:
            return None
        return self._rng.choice(candidates)

    def set_entry_elo(self, entry_name: str, elo: float) -> bool:
        for entry in self.entries:
            if entry.name == entry_name:
                entry.elo = elo
                self._save()
                return True
        return False

    def _next_version_index(self) -> int:
        max_idx = 0
        for entry in self.entries:
            if entry.name.startswith("model_v"):
                suffix = entry.name.replace("model_v", "")
                if suffix.isdigit():
                    max_idx = max(max_idx, int(suffix))
        return max_idx + 1

    def _clear_best_flag(self) -> None:
        for entry in self.entries:
            entry.is_best = False

    def _trim_versions(self) -> None:
        if len(self.entries) <= self.max_versions:
            return
        # Keep latest versions and always keep current best entry.
        best = self.get_best()
        sorted_entries = sorted(self.entries, key=lambda e: e.step, reverse=True)
        kept: List[ModelEntry] = []
        for entry in sorted_entries:
            if len(kept) < self.max_versions:
                kept.append(entry)
                continue
            if best is not None and entry.name == best.name and best.name not in {k.name for k in kept}:
                replaced = kept[-1]
                kept[-1] = entry
                if Path(replaced.path).exists():
                    Path(replaced.path).unlink(missing_ok=True)
            else:
                if Path(entry.path).exists():
                    Path(entry.path).unlink(missing_ok=True)
        self.entries = sorted(kept, key=lambda e: e.step)
