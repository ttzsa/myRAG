# Regression tests for the MinerU CLI batch runner wrapper.
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from offline_index.document_manifest import load_manifest, save_manifest, upsert_document
from offline_index.mineru_runner import (
    expected_mineru_output_dir,
    find_pdfs,
    is_already_parsed,
    parse_pdf_candidates,
    parse_pdf_dir,
    run_mineru_cli,
)
from offline_index.source_file_finder import compute_file_md5, scan_pdfs
from offline_index.schema import DocumentRecord, PdfCandidate


class MinerURunnerTests(unittest.TestCase):
    """Covers PDF discovery, skip detection, subprocess command construction, and batch resilience."""

    def test_find_pdfs_accepts_file_and_directory_with_recursive_option(self):
        """Verify PDF scanning handles a single file, recursive directories, and non-recursive directories."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            top_pdf = root / "A Paper.pdf"
            nested_dir = root / "nested"
            nested_pdf = nested_dir / "B.pdf"
            nested_dir.mkdir()
            top_pdf.write_bytes(b"%PDF")
            nested_pdf.write_bytes(b"%PDF")
            (root / "note.txt").write_text("ignore", encoding="utf-8")
            (root / "~$temp.pdf").write_bytes(b"%PDF")
            (root / ".hidden.pdf").write_bytes(b"%PDF")

            self.assertEqual(find_pdfs(top_pdf), [top_pdf.resolve()])
            self.assertEqual(find_pdfs(root, recursive=False), [top_pdf.resolve()])
            self.assertEqual(find_pdfs(root, recursive=True), [top_pdf.resolve(), nested_pdf.resolve()])

    def test_expected_output_and_skip_detection_follow_mineru_auto_dir(self):
        """Verify parsed outputs are detected under output_root/pdf_stem/auto."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "A Paper.pdf"
            output_root = root / "out"
            pdf.write_bytes(b"%PDF")
            auto_dir = output_root / "A Paper" / "auto"

            self.assertEqual(expected_mineru_output_dir(pdf, output_root), auto_dir.resolve())
            self.assertFalse(is_already_parsed(pdf, output_root))

            auto_dir.mkdir(parents=True)
            (auto_dir / "A Paper_content_list_v2.json").write_text("[]", encoding="utf-8")

            self.assertTrue(is_already_parsed(pdf, output_root))

    def test_run_mineru_cli_parses_when_output_exists_but_md5_missing_from_manifest(self):
        """Verify output files alone no longer cause a skip when the PDF MD5 is not in manifest."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "A.pdf"
            output_root = root / "out"
            mineru = root / "mineru.exe"
            manifest_path = root / "processed_pdfs.json"
            pdf.write_bytes(b"%PDF")
            mineru.write_bytes(b"exe")
            auto_dir = output_root / "A" / "auto"
            auto_dir.mkdir(parents=True)
            (auto_dir / "A_content_list_v2.json").write_text("[]", encoding="utf-8")
            completed = Mock(returncode=0, stdout="ok", stderr="")

            with patch("offline_index.mineru_runner.subprocess.run", return_value=completed) as run:
                result = run_mineru_cli(pdf, output_root, mineru, manifest_path=manifest_path)

            run.assert_called_once()
            self.assertTrue(result.success)
            self.assertFalse(result.skipped)
            self.assertEqual(result.return_code, 0)
            manifest = load_manifest(manifest_path)
            self.assertEqual(len(manifest.documents), 1)
            self.assertEqual(manifest.documents[0].pdf_md5, compute_file_md5(pdf))

    def test_run_mineru_cli_skips_when_manifest_md5_and_content_list_v2_exist(self):
        """Verify the same PDF content is skipped only when manifest and v2 output are present."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "A.pdf"
            output_root = root / "out"
            mineru = root / "mineru.exe"
            manifest_path = root / "processed_pdfs.json"
            pdf.write_bytes(b"%PDF")
            mineru.write_bytes(b"exe")
            auto_dir = output_root / "A" / "auto"
            auto_dir.mkdir(parents=True)
            (auto_dir / "A_content_list_v2.json").write_text("[]", encoding="utf-8")
            manifest = load_manifest(manifest_path)
            upsert_document(
                manifest,
                DocumentRecord(
                    doc_id="doc_a",
                    file_name=pdf.name,
                    source_path=str(pdf.resolve()),
                    pdf_md5=compute_file_md5(pdf),
                    file_size=pdf.stat().st_size,
                    mineru_output_dir=str(auto_dir.resolve()),
                    index_status="parsed",
                ),
            )
            save_manifest(manifest, manifest_path)

            with patch("offline_index.mineru_runner.subprocess.run") as run:
                result = run_mineru_cli(pdf, output_root, mineru, manifest_path=manifest_path)

            run.assert_not_called()
            self.assertTrue(result.success)
            self.assertTrue(result.skipped)
            self.assertEqual(result.expected_output_dir, str(auto_dir.resolve()))

    def test_run_mineru_cli_reparses_when_manifest_md5_exists_but_v2_missing(self):
        """Verify a known MD5 is reparsed when its recorded content_list_v2 output is missing."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "A.pdf"
            output_root = root / "out"
            mineru = root / "mineru.exe"
            manifest_path = root / "processed_pdfs.json"
            pdf.write_bytes(b"%PDF")
            mineru.write_bytes(b"exe")
            auto_dir = output_root / "A" / "auto"
            auto_dir.mkdir(parents=True)
            manifest = load_manifest(manifest_path)
            upsert_document(
                manifest,
                DocumentRecord(
                    doc_id="doc_a",
                    file_name=pdf.name,
                    source_path=str(pdf.resolve()),
                    pdf_md5=compute_file_md5(pdf),
                    file_size=pdf.stat().st_size,
                    mineru_output_dir=str(auto_dir.resolve()),
                    index_status="parsed",
                ),
            )
            save_manifest(manifest, manifest_path)
            completed = Mock(returncode=0, stdout="ok", stderr="")

            with patch("offline_index.mineru_runner.subprocess.run", return_value=completed) as run:
                result = run_mineru_cli(pdf, output_root, mineru, manifest_path=manifest_path)

            run.assert_called_once()
            self.assertTrue(result.success)
            self.assertFalse(result.skipped)

    def test_run_mineru_cli_reparses_when_same_path_content_changes(self):
        """Verify replacing a PDF at the same path changes MD5 and triggers parsing despite old output."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "A.pdf"
            output_root = root / "out"
            mineru = root / "mineru.exe"
            manifest_path = root / "processed_pdfs.json"
            pdf.write_bytes(b"old")
            mineru.write_bytes(b"exe")
            auto_dir = output_root / "A" / "auto"
            auto_dir.mkdir(parents=True)
            (auto_dir / "A_content_list_v2.json").write_text("[]", encoding="utf-8")
            old_md5 = compute_file_md5(pdf)
            manifest = load_manifest(manifest_path)
            upsert_document(
                manifest,
                DocumentRecord(
                    doc_id="doc_a",
                    file_name=pdf.name,
                    source_path=str(pdf.resolve()),
                    pdf_md5=old_md5,
                    file_size=pdf.stat().st_size,
                    mineru_output_dir=str(auto_dir.resolve()),
                    index_status="parsed",
                ),
            )
            save_manifest(manifest, manifest_path)
            pdf.write_bytes(b"new")
            completed = Mock(returncode=0, stdout="ok", stderr="")

            with patch("offline_index.mineru_runner.subprocess.run", return_value=completed) as run:
                result = run_mineru_cli(pdf, output_root, mineru, manifest_path=manifest_path)

            run.assert_called_once()
            self.assertTrue(result.success)
            manifest = load_manifest(manifest_path)
            self.assertEqual(len(manifest.documents), 1)
            self.assertEqual(manifest.documents[0].pdf_md5, compute_file_md5(pdf))
            self.assertNotEqual(manifest.documents[0].pdf_md5, old_md5)

    def test_run_mineru_cli_skips_different_path_with_same_md5(self):
        """Verify duplicate PDF content at a different path reuses the existing manifest record."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "A.pdf"
            second = root / "B.pdf"
            output_root = root / "out"
            mineru = root / "mineru.exe"
            manifest_path = root / "processed_pdfs.json"
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            mineru.write_bytes(b"exe")
            auto_dir = output_root / "A" / "auto"
            auto_dir.mkdir(parents=True)
            (auto_dir / "A_content_list_v2.json").write_text("[]", encoding="utf-8")
            manifest = load_manifest(manifest_path)
            upsert_document(
                manifest,
                DocumentRecord(
                    doc_id="doc_a",
                    file_name=first.name,
                    source_path=str(first.resolve()),
                    pdf_md5=compute_file_md5(first),
                    file_size=first.stat().st_size,
                    mineru_output_dir=str(auto_dir.resolve()),
                    index_status="parsed",
                ),
            )
            save_manifest(manifest, manifest_path)

            with patch("offline_index.mineru_runner.subprocess.run") as run:
                result = run_mineru_cli(second, output_root, mineru, manifest_path=manifest_path)

            run.assert_not_called()
            self.assertTrue(result.skipped)
            manifest = load_manifest(manifest_path)
            self.assertEqual(len(manifest.documents), 1)
            self.assertEqual(manifest.documents[0].source_path, str(first.resolve()))

    def test_run_mineru_cli_builds_pipeline_command_without_shell(self):
        """Verify the CLI call uses subprocess.run with list args, pipeline backend, and method."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "A Paper.pdf"
            output_root = root / "out"
            mineru = root / "mineru.exe"
            pdf.write_bytes(b"%PDF")
            mineru.write_bytes(b"exe")
            completed = Mock(returncode=0, stdout="ok", stderr="")

            with patch("offline_index.mineru_runner.subprocess.run", return_value=completed) as run:
                result = run_mineru_cli(pdf, output_root, mineru, force=True, timeout=30)

            command = run.call_args.args[0]
            options = run.call_args.kwargs
            self.assertEqual(command, [str(mineru.resolve()), "-p", str(pdf.resolve()), "-o", str(output_root.resolve()), "-b", "pipeline", "-m", "auto"])
            self.assertFalse(options.get("shell", False))
            self.assertTrue(options["capture_output"])
            self.assertTrue(options["text"])
            self.assertEqual(options["encoding"], "utf-8")
            self.assertEqual(options["errors"], "replace")
            self.assertEqual(options["timeout"], 30)
            self.assertTrue(result.success)
            self.assertFalse(result.skipped)

    def test_run_mineru_cli_passes_local_model_env_to_subprocess(self):
        """Verify MinerU local model settings are passed into the child process environment."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "A Paper.pdf"
            output_root = root / "out"
            mineru = root / "mineru.exe"
            tools_config = root / "mineru.json"
            pdf.write_bytes(b"%PDF")
            mineru.write_bytes(b"exe")
            tools_config.write_text("{}", encoding="utf-8")
            completed = Mock(returncode=0, stdout="ok", stderr="")

            with patch("offline_index.mineru_runner.subprocess.run", return_value=completed) as run:
                result = run_mineru_cli(
                    pdf,
                    output_root,
                    mineru,
                    force=True,
                    model_source="local",
                    tools_config_json=tools_config,
                )

            options = run.call_args.kwargs
            self.assertEqual(options["env"]["MINERU_MODEL_SOURCE"], "local")
            self.assertEqual(options["env"]["MINERU_TOOLS_CONFIG_JSON"], str(tools_config.resolve()))
            self.assertTrue(result.success)
            self.assertFalse(result.skipped)


    def test_parse_pdf_dir_continues_after_one_failure(self):
        """Verify batch parsing records failures and continues with later PDFs."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "out"
            first = root / "first.pdf"
            second = root / "second.pdf"
            mineru = root / "mineru.exe"
            first.write_bytes(b"%PDF")
            second.write_bytes(b"%PDF")
            mineru.write_bytes(b"exe")
            failed = Mock(returncode=2, stdout="", stderr="failed")
            succeeded = Mock(returncode=0, stdout="ok", stderr="")

            with patch("offline_index.mineru_runner.subprocess.run", side_effect=[failed, succeeded]):
                results = parse_pdf_dir(root, output_root, mineru, recursive=False, force=True)

            self.assertEqual(len(results), 2)
            self.assertFalse(results[0].success)
            self.assertIn("failed", results[0].error_message)
            self.assertTrue(results[1].success)

    def test_parse_pdf_candidates_parses_only_discovered_selection(self):
        """Verify parse stage processes exactly the candidates selected by discover."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "out"
            first = root / "first.pdf"
            second = root / "second.pdf"
            mineru = root / "mineru.exe"
            first.write_bytes(b"%PDF first")
            second.write_bytes(b"%PDF second")
            mineru.write_bytes(b"exe")
            candidates = scan_pdfs(root, recursive=False)
            selected = [candidate for candidate in candidates if candidate.file_name == "second.pdf"]
            completed = Mock(returncode=0, stdout="ok", stderr="")

            with patch("offline_index.mineru_runner.subprocess.run", return_value=completed) as run:
                results = parse_pdf_candidates(selected, output_root, mineru, parser_method="mineru")

            self.assertEqual([Path(result.pdf_path).name for result in results], ["second.pdf"])
            self.assertEqual(run.call_count, 1)
            self.assertIn(str(second.resolve()), run.call_args.args[0])

    def test_parse_pdf_candidates_rejects_unimplemented_parser_method(self):
        """Verify parser-method is explicit when a parser is not available yet."""

        candidate = PdfCandidate(
            source_path=str(Path("paper.pdf").resolve()),
            file_name="paper.pdf",
            file_size=1,
            pdf_md5="abc",
            modified_time=1.0,
        )

        with self.assertRaisesRegex(ValueError, "pymupdf"):
            parse_pdf_candidates([candidate], Path("out"), Path("parser.exe"), parser_method="pymupdf")

    def test_run_mineru_cli_error_message_keeps_full_output(self):
        """Verify failed MinerU calls keep full stderr/stdout details."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "A.pdf"
            output_root = root / "out"
            mineru = root / "mineru.exe"
            pdf.write_bytes(b"%PDF")
            mineru.write_bytes(b"exe")
            stderr = "INFO started local api\nERROR parse failed\nTraceback line"
            completed = Mock(returncode=2, stdout="stdout fallback", stderr=stderr)

            with patch("offline_index.mineru_runner.subprocess.run", return_value=completed):
                result = run_mineru_cli(pdf, output_root, mineru, force=True)

            self.assertEqual(result.error_message, stderr)

    def test_parse_pdf_dir_excludes_output_root_when_nested_under_input(self):
        """Verify recursive scans do not parse MinerU-generated PDFs inside output_root."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "output_pipeline"
            source_pdf = root / "source.pdf"
            generated_pdf = output_root / "source" / "auto" / "source_origin.pdf"
            mineru = root / "mineru.exe"
            generated_pdf.parent.mkdir(parents=True)
            source_pdf.write_bytes(b"%PDF")
            generated_pdf.write_bytes(b"%PDF")
            mineru.write_bytes(b"exe")
            completed = Mock(returncode=0, stdout="ok", stderr="")

            with patch("offline_index.mineru_runner.subprocess.run", return_value=completed) as run:
                results = parse_pdf_dir(root, output_root, mineru, recursive=True, force=True)

            self.assertEqual(len(results), 1)
            self.assertEqual(Path(results[0].pdf_path), source_pdf.resolve())
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
