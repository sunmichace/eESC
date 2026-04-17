from __future__ import annotations

"""Project-level paths and task metadata."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
ANALYSIS_REPORT_DIR = REPORTS_DIR / "analysis_results"
MPLCONFIG_DIR = PROJECT_ROOT / ".matplotlib"


@dataclass(frozen=True)
class TaskConfig:
    key: str
    directory_name: str
    display_name: str
    description: str
    data_directory_name: str | None = None

    @property
    def task_root(self) -> Path:
        return PROJECT_ROOT / "tasks" / self.directory_name

    @property
    def code_dir(self) -> Path:
        return self.task_root / "code"

    @property
    def best_model_dir(self) -> Path:
        return self.code_dir / "best_model_save"

    @property
    def machine_log_path(self) -> Path:
        return self.best_model_dir / "machine_results" / "train_log.txt"


TASKS = {
    config.key: config
    for config in (
        TaskConfig(
            key="kcat",
            directory_name="kcat_tc",
            display_name="kcat",
            description="Single-task kcat regression experiment.",
            data_directory_name="kcat",
        ),
        TaskConfig(
            key="km",
            directory_name="km_tc",
            display_name="Km",
            description="Single-task Km regression experiment.",
            data_directory_name="km",
        ),
        TaskConfig(
            key="kcatkm",
            directory_name="kcatkm_tc",
            display_name="kcat/km",
            description="Joint or extended-task experiment branch.",
            data_directory_name="kcatkm",
        ),
    )
}


def configure_matplotlib_env() -> None:
    """Keep matplotlib cache local to the repository."""
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

