import json
import re
import subprocess
from pathlib import Path


def _first_line_description(line: str) -> str:
    match = re.match(r"#\s*Cell\s+\d+\s*:?\s*(.*)", line)
    if match:
        return match.group(1).strip()
    return line.lstrip("# ").strip()


def test_tracked_notebooks_use_code_only_numbered_cell_headers():
    tracked = subprocess.check_output(["git", "ls-files", "notebooks/*.ipynb"], text=True)
    notebook_paths = [Path(line) for line in tracked.splitlines()]

    assert notebook_paths, "Expected tracked notebooks to exist."

    for notebook_path in notebook_paths:
        data = json.loads(notebook_path.read_text(encoding="utf-8"))
        for idx, cell in enumerate(data.get("cells", []), start=1):
            assert cell.get("cell_type") == "code", f"{notebook_path} cell {idx} is not code"
            source = "".join(cell.get("source", []))
            assert source.startswith(
                f"# Cell {idx} : "
            ), f"{notebook_path} cell {idx} does not use required header"


def test_tracked_notebooks_keep_main_cell_order_and_notes():
    tracked = subprocess.check_output(["git", "ls-files", "notebooks/*.ipynb"], text=True)
    for notebook in tracked.splitlines():
        main_text = subprocess.check_output(["git", "show", f"origin/main:{notebook}"], text=True)
        main = json.loads(main_text)
        current = json.loads(Path(notebook).read_text(encoding="utf-8"))

        assert len(current.get("cells", [])) == len(main.get("cells", [])), notebook
        for idx, (main_cell, current_cell) in enumerate(
            zip(main.get("cells", []), current.get("cells", []), strict=True),
            start=1,
        ):
            main_source = "".join(main_cell.get("source", [])).splitlines()
            current_source = "".join(current_cell.get("source", [])).splitlines()
            main_desc = _first_line_description(main_source[0]) if main_source else ""
            current_desc = _first_line_description(current_source[0]) if current_source else ""
            assert current_desc == main_desc, f"{notebook} cell {idx}"
