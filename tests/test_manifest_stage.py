# Regression tests for PDF scanning, MinerU output location, and JSON manifest maintenance.
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from offline_index.document_manifest import load_manifest, save_manifest, upsert_document
from offline_index.index_report import IndexReport
from offline_index.mineru_output_locator import locate_mineru_output
from offline_index.source_file_finder import compute_file_md5, scan_pdfs
from offline_index.schema import ChunkRecord, DocumentRecord, PdfCandidate
from scripts.build_manifest import _format_elapsed, _make_preview_file_name, _print_chunk_progress, _process_candidate


class ManifestStageTests(unittest.TestCase):
    """Covers second-stage document scanning and manifest behavior."""

    def test_scan_pdfs_returns_candidates_with_content_md5_and_skips_temp_files(self):
        """Verify PDF scanning is recursive, content-hash based, and ignores temp/non-PDF files."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            pdf = root / "paper.pdf"
            nested_pdf = nested / "other.PDF"
            temp_pdf = root / "~$temp.pdf"
            pdf.write_bytes(b"same content")
            nested_pdf.write_bytes(b"other content")
            temp_pdf.write_bytes(b"temp")
            (root / "note.txt").write_text("ignore", encoding="utf-8")

            candidates = scan_pdfs(root, recursive=True)

            self.assertEqual([candidate.file_name for candidate in candidates], ["paper.pdf", "other.PDF"])
            self.assertEqual(candidates[0].pdf_md5, compute_file_md5(pdf))
            self.assertEqual(candidates[0].file_size, len(b"same content"))
            self.assertGreater(candidates[0].modified_time, 0)

    def test_manifest_load_save_find_and_upsert(self):
        """Verify manifest creation, lookup by MD5/source path, and record replacement."""

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "data" / "index" / "rag_documents.json"
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest.documents, [])

            record = DocumentRecord(
                doc_id="doc_1",
                file_name="paper.pdf",
                source_path=str((Path(tmp) / "paper.pdf").resolve()),
                pdf_md5="abc",
                file_size=12,
                mineru_output_dir="out/paper/auto",
                index_status="parsed",
            )
            upsert_document(manifest, record)
            self.assertEqual(manifest.find_by_md5("abc").doc_id, "doc_1")
            self.assertEqual(manifest.find_by_source_path(record.source_path).doc_id, "doc_1")

            updated = record.model_copy(update={"index_status": "chunked", "chunk_count": 3})
            upsert_document(manifest, updated)
            save_manifest(manifest, manifest_path)
            reloaded = load_manifest(manifest_path)

            self.assertEqual(len(reloaded.documents), 1)
            self.assertEqual(reloaded.documents[0].index_status, "chunked")
            self.assertEqual(reloaded.documents[0].chunk_count, 3)

    def test_locate_mineru_output_reports_paths_and_missing_v2(self):
        """Verify MinerU output locator handles spaces, success, warnings, and missing output."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "A Paper.pdf"
            output_root = root / "output"
            auto_dir = output_root / "A Paper" / "auto"
            auto_dir.mkdir(parents=True)
            pdf.write_bytes(b"%PDF")
            (auto_dir / "A Paper_content_list_v2.json").write_text("[]", encoding="utf-8")
            (auto_dir / "A Paper_content_list.json").write_text("[]", encoding="utf-8")
            (auto_dir / "A Paper.md").write_text("# ok", encoding="utf-8")

            located = locate_mineru_output(pdf, output_root)

            self.assertTrue(located.success)
            self.assertEqual(Path(located.mineru_output_dir), auto_dir.resolve())
            self.assertEqual(Path(located.content_list_v2_path), (auto_dir / "A Paper_content_list_v2.json").resolve())
            self.assertIn("images directory not found", located.warning)

            (auto_dir / "A Paper_content_list_v2.json").unlink()
            missing = locate_mineru_output(pdf, output_root)

            self.assertFalse(missing.success)
            self.assertIn("content_list_v2", missing.error_message)

    def test_manifest_json_shape_contains_documents_key(self):
        """Verify saved manifest uses the expected top-level documents list."""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rag_documents.json"
            manifest = load_manifest(path)
            upsert_document(
                manifest,
                DocumentRecord(
                    doc_id="doc_1",
                    file_name="paper.pdf",
                    source_path="paper.pdf",
                    pdf_md5="abc",
                    file_size=1,
                ),
            )

            save_manifest(manifest, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(list(payload.keys()), ["documents"])
            self.assertEqual(payload["documents"][0]["doc_id"], "doc_1")

    def test_build_manifest_does_not_skip_existing_md5_when_content_list_v2_missing(self):
        """Verify an existing MD5 with missing v2 output is marked pending instead of skipped."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "paper.pdf"
            output_root = root / "out"
            missing_auto = output_root / "paper" / "auto"
            preview_dir = root / "debug"
            pdf.write_bytes(b"%PDF")
            missing_auto.mkdir(parents=True)
            candidate = PdfCandidate(
                source_path=str(pdf.resolve()),
                file_name=pdf.name,
                file_size=pdf.stat().st_size,
                pdf_md5=compute_file_md5(pdf),
                modified_time=pdf.stat().st_mtime,
            )
            manifest = load_manifest(root / "rag_documents.json")
            upsert_document(
                manifest,
                DocumentRecord(
                    doc_id="doc_1",
                    file_name=pdf.name,
                    source_path=str(pdf.resolve()),
                    pdf_md5=candidate.pdf_md5,
                    file_size=candidate.file_size,
                    mineru_output_dir=str(missing_auto.resolve()),
                    index_status="parsed",
                ),
            )
            args = Namespace(
                force=False,
                mineru_output_root=output_root,
                build_chunks_preview=False,
                preview_output_dir=preview_dir,
                chunk_size=800,
                chunk_overlap=120,
            )
            report = IndexReport()

            _process_candidate(candidate, args, manifest, report)

            self.assertEqual(report.existing_skipped, 0)
            self.assertEqual(report.mineru_output_missing, 1)
            self.assertEqual(manifest.documents[0].index_status, "pending")
            self.assertIn("content_list_v2", manifest.documents[0].error_message)

    def test_build_manifest_builds_preview_for_existing_md5_without_chunks(self):
        """Verify parsed documents without chunks are not skipped when preview build is requested."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "HVI.pdf"
            output_root = root / "out"
            auto_dir = output_root / "HVI" / "auto"
            preview_dir = root / "debug"
            pdf.write_bytes(b"%PDF")
            auto_dir.mkdir(parents=True)
            (auto_dir / "HVI_content_list_v2.json").write_text("[]", encoding="utf-8")
            candidate = PdfCandidate(
                source_path=str(pdf.resolve()),
                file_name=pdf.name,
                file_size=pdf.stat().st_size,
                pdf_md5=compute_file_md5(pdf),
                modified_time=pdf.stat().st_mtime,
            )
            manifest = load_manifest(root / "rag_documents.json")
            upsert_document(
                manifest,
                DocumentRecord(
                    doc_id="doc_hvi",
                    file_name=pdf.name,
                    source_path=str(pdf.resolve()),
                    pdf_md5=candidate.pdf_md5,
                    file_size=candidate.file_size,
                    mineru_output_dir=str(auto_dir.resolve()),
                    index_status="parsed",
                    chunk_count=0,
                ),
            )
            args = Namespace(
                force=False,
                mineru_output_root=output_root,
                build_chunks_preview=True,
                preview_output_dir=preview_dir,
                chunk_size=800,
                chunk_overlap=120,
            )
            report = IndexReport()

            with patch("scripts.build_manifest._build_chunks_preview", return_value=(Counter({"text": 1}), preview_dir / "HVI_chunks_preview.json", None)) as build:
                _process_candidate(candidate, args, manifest, report)

            build.assert_called_once()
            self.assertEqual(report.existing_skipped, 0)
            self.assertEqual(report.chunk_preview_generated, 1)
            self.assertEqual(manifest.documents[0].index_status, "chunked")
            self.assertEqual(manifest.documents[0].chunk_count, 1)

    def test_build_manifest_skips_existing_md5_with_chunks_when_not_forced(self):
        """Verify already chunked documents still skip preview rebuilding without --force."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "paper.pdf"
            output_root = root / "out"
            auto_dir = output_root / "paper" / "auto"
            preview_dir = root / "debug"
            pdf.write_bytes(b"%PDF")
            auto_dir.mkdir(parents=True)
            (auto_dir / "paper_content_list_v2.json").write_text("[]", encoding="utf-8")
            candidate = PdfCandidate(
                source_path=str(pdf.resolve()),
                file_name=pdf.name,
                file_size=pdf.stat().st_size,
                pdf_md5=compute_file_md5(pdf),
                modified_time=pdf.stat().st_mtime,
            )
            manifest = load_manifest(root / "rag_documents.json")
            upsert_document(
                manifest,
                DocumentRecord(
                    doc_id="doc_1",
                    file_name=pdf.name,
                    source_path=str(pdf.resolve()),
                    pdf_md5=candidate.pdf_md5,
                    file_size=candidate.file_size,
                    mineru_output_dir=str(auto_dir.resolve()),
                    index_status="chunked",
                    chunk_count=1,
                    text_chunk_count=1,
                ),
            )
            args = Namespace(
                force=False,
                mineru_output_root=output_root,
                build_chunks_preview=True,
                preview_output_dir=preview_dir,
                chunk_size=800,
                chunk_overlap=120,
            )
            report = IndexReport()

            with patch("scripts.build_manifest._build_chunks_preview") as build:
                _process_candidate(candidate, args, manifest, report)

            build.assert_not_called()
            self.assertEqual(report.existing_skipped, 1)
            self.assertEqual(report.chunk_preview_generated, 0)

    def test_preview_file_name_uses_pdf_stem_and_full_md5(self):
        """Verify chunk preview files are named from the source PDF stem and full MD5."""

        candidate = PdfCandidate(
            source_path=str(Path("pdfs") / "A Paper.pdf"),
            file_name="A Paper.pdf",
            file_size=10,
            pdf_md5="0123456789abcdef0123456789abcdef",
            modified_time=1.0,
        )

        self.assertEqual(
            _make_preview_file_name(candidate),
            "A Paper_0123456789abcdef0123456789abcdef_chunks_preview.json",
        )

    def test_format_elapsed_uses_minute_and_second_display(self):
        """Verify elapsed time display supports short and longer durations."""

        self.assertEqual(_format_elapsed(9.876), "9.88s")
        self.assertEqual(_format_elapsed(125.432), "2m 5.43s")

    def test_print_chunk_progress_includes_document_and_chunk_position(self):
        """Verify chunk progress reports the current document and chunk identity."""

        chunks = [
            ChunkRecord(
                id="chunk_text",
                document="text",
                metadata={"chunk_type": "text", "page_start": 1, "page_end": 1},
            ),
            ChunkRecord(
                id="chunk_image",
                document="image",
                metadata={"chunk_type": "image", "page_start": 2, "page_end": 3},
            ),
        ]
        output = StringIO()

        with redirect_stdout(output):
            _print_chunk_progress("paper.pdf", chunks)

        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], "processing chunk: paper.pdf [1/2] text chunk_text pages 1")
        self.assertEqual(lines[1], "processing chunk: paper.pdf [2/2] image chunk_image pages 2-3")


if __name__ == "__main__":
    unittest.main()
