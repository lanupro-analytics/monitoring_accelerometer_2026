"""Scan notebooks and Quarto/R Markdown documents in this folder for Python
imports, then write a pinned requirements.txt at the project root.

Run with:
    python 99_update_requirements.py
"""

import ast
import json
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path

THIS_SCRIPT = "99_update_requirements.py"
PYTHON_CHUNK_RE = re.compile(
    r"^[ \t]*```\{?python[^\n}]*\}?[ \t]*\n(.*?)^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def python_sources(path: Path) -> list[str]:
    """Return each chunk/cell of Python source code found in a notebook or qmd/Rmd file."""
    if path.suffix.lower() == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
        return [
            "\n".join(cell.get("source", []))
            for cell in cells
            if cell.get("cell_type") == "code"
        ]
    return PYTHON_CHUNK_RE.findall(path.read_text(encoding="utf-8"))


def top_level_imports(code: str) -> set[str]:
    """Return top-level module names imported in a snippet of Python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def distribution_for(module: str, module_to_dists: dict) -> str | None:
    """Map an imported module name to an installed distribution name, if any."""
    dists = module_to_dists.get(module)
    if dists:
        return sorted(dists)[0]
    try:
        version(module)  # fallback: module name often matches distribution name
        return module
    except PackageNotFoundError:
        return None


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    notebooks_dir = root / "notebooks_code"
    requirements_txt = root / "requirements.txt"

    # Discover source files, skipping checkpoints and this script itself
    # (its own source contains "read_excel(" / ".xlsx" strings that would otherwise
    # be misdetected as real usage).
    source_files = sorted(
        p for p in notebooks_dir.rglob("*")
        if p.suffix.lower() in (".ipynb", ".qmd", ".rmd")
        and "ipynb_checkpoints" not in p.parts
        and p.name != THIS_SCRIPT
    )

    # Scan all source files for imports and for pandas Excel usage requiring openpyxl.
    used_modules = set()
    needs_openpyxl = False
    for path in source_files:
        for src in python_sources(path):
            used_modules.update(top_level_imports(src))
            if "read_excel(" in src and ".xlsx" in src:
                needs_openpyxl = True

    # Resolve modules to installed, pinned distributions.
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    module_to_dists = packages_distributions()
    third_party = sorted(m for m in used_modules if m not in stdlib)

    packages = set()
    unresolved_modules = []
    for m in third_party:
        dist_name = distribution_for(m, module_to_dists)
        if dist_name is None:
            unresolved_modules.append(m)
        else:
            packages.add(dist_name)

    if needs_openpyxl:
        packages.add("openpyxl")

    pinned_requirements = []
    missing_versions = []
    for pkg in sorted(packages):
        try:
            pinned_requirements.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            missing_versions.append(pkg)

    if missing_versions:
        raise RuntimeError(
            f"Cannot pin versions for: {', '.join(missing_versions)}. "
            "Install them first, then rerun this script."
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    python_version = ".".join(map(str, sys.version_info[:3]))
    header = [
        "# Auto-generated from notebook/qmd/rmd imports",
        "# Source files:",
        *(f"# - {p.relative_to(root).as_posix()}" for p in source_files),
        f"# Generated at: {generated_at}",
        f"# Python version: {python_version}",
        "",
    ]
    requirements_txt.write_text(
        "\n".join(header + pinned_requirements) + "\n", encoding="utf-8"
    )

    print("Files scanned:")
    for p in source_files:
        print(f"- {p}")

    if unresolved_modules:
        print("\nWarning: unresolved third-party modules (skipped):")
        for m in unresolved_modules:
            print(f"- {m}")

    print("\nPinned packages written to requirements.txt:")
    for req in pinned_requirements:
        print(f"- {req}")

    print(f"\nUpdated: {requirements_txt}")


if __name__ == "__main__":
    main()
