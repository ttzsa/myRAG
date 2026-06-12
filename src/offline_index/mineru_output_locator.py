# Locates existing MinerU output files for a source PDF without invoking MinerU.
from __future__ import annotations

from pathlib import Path

from offline_index.schema import MinerUOutputLocation


def locate_mineru_output(pdf_path: Path, mineru_output_root: Path) -> MinerUOutputLocation:
    """Locate expected MinerU output paths for a source PDF."""

    output_dir = (mineru_output_root.resolve() / pdf_path.stem / "auto").resolve()
    content_list_v2 = _first_match(output_dir, "*_content_list_v2.json")
    content_list = _first_match(output_dir, "*_content_list.json")
    markdown = _first_match(output_dir, "*.md")
    images_dir = output_dir / "images"
    warning = ""

    if not output_dir.is_dir():
        return MinerUOutputLocation(
            mineru_output_dir=str(output_dir),
            success=False,
            error_message=f"MinerU output directory not found: {output_dir}",
        )
    if content_list_v2 is None:
        return MinerUOutputLocation(
            mineru_output_dir=str(output_dir),
            content_list_path=str(content_list) if content_list else "",
            markdown_path=str(markdown) if markdown else "",
            images_dir=str(images_dir) if images_dir.is_dir() else "",
            success=False,
            warning="images directory not found" if not images_dir.is_dir() else "",
            error_message=f"content_list_v2.json not found under: {output_dir}",
        )
    if not images_dir.is_dir():
        warning = "images directory not found"

    return MinerUOutputLocation(
        mineru_output_dir=str(output_dir),
        content_list_v2_path=str(content_list_v2),
        content_list_path=str(content_list) if content_list else "",
        markdown_path=str(markdown) if markdown else "",
        images_dir=str(images_dir) if images_dir.is_dir() else "",
        success=True,
        warning=warning,
    )


def _first_match(directory: Path, pattern: str) -> Path | None:
    """Return the first matching file under a directory, or None."""

    if not directory.is_dir():
        return None
    matches = sorted(path.resolve() for path in directory.glob(pattern) if path.is_file())
    return matches[0] if matches else None

