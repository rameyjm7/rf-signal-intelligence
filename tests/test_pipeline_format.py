import ast
import re
import subprocess
from pathlib import Path


def _first_line_description(line: str) -> str:
    match = re.match(r"#\s*Cell\s+\d+\s*:?\s*(.*)", line)
    if match:
        return match.group(1).strip()
    return line.lstrip("# ").strip()


def test_tracked_pipelines_are_valid_python():
    tracked = subprocess.check_output(["git", "ls-files", "pipelines/*.py"], text=True)
    pipeline_paths = [Path(line) for line in tracked.splitlines()]

    assert pipeline_paths, "Expected tracked pipeline scripts to exist."

    for pipeline_path in pipeline_paths:
        ast.parse(pipeline_path.read_text(encoding="utf-8"), filename=str(pipeline_path))


def test_tracked_pipelines_use_unique_cell_descriptions():
    tracked = subprocess.check_output(["git", "ls-files", "pipelines/*.py"], text=True)
    for pipeline in tracked.splitlines():
        source = Path(pipeline).read_text(encoding="utf-8").splitlines()
        descriptions = [
            _first_line_description(line)
            for line in source
            if line.startswith("# Cell ") and " : " in line
        ]

        assert descriptions, f"{pipeline} has no converted cell descriptions"
        assert len(descriptions) == len(set(descriptions)), f"{pipeline} has duplicate cell descriptions"
