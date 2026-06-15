# Regression tests for the unified offline workflow CLI command planning.
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.build_index import _collect_chunk_paths
from scripts.offline import build_stage_commands, parse_args


class OfflineCliTests(unittest.TestCase):
    """Covers command construction for the four-stage offline workflow."""

    def test_default_workflow_runs_parse_chunk_and_index(self):
        """Verify default offline flow uses new PDFs, MinerU parsing, auto VLM, and no reset."""

        args = parse_args([])
        commands = build_stage_commands(args)

        self.assertEqual(len(commands), 3)
        self.assertEqual(Path(commands[0][1]).name, "parse_pdfs.py")
        self.assertIn("--pdf-scope", commands[0])
        self.assertIn("new", commands[0])
        self.assertIn("--parser-method", commands[0])
        self.assertIn("mineru", commands[0])
        self.assertEqual(Path(commands[1][1]).name, "build_manifest.py")
        self.assertIn("--vlm-mode", commands[1])
        self.assertIn("auto", commands[1])
        self.assertEqual(Path(commands[2][1]).name, "build_index.py")
        self.assertNotIn("--reset", commands[2])

    def test_chunk_subcommand_accepts_multiple_content_list_v2_paths(self):
        """Verify chunk subcommand forwards explicit content_list_v2 selections."""

        args = parse_args(["chunk", "--content-list-v2", "a_content_list_v2.json", "--content-list-v2", "b_content_list_v2.json", "--vlm-mode", "refresh"])
        commands = build_stage_commands(args)

        self.assertEqual(len(commands), 1)
        self.assertEqual(Path(commands[0][1]).name, "build_manifest.py")
        self.assertEqual(commands[0].count("--content-list-v2"), 2)
        self.assertIn("refresh", commands[0])

    def test_index_subcommand_forwards_reset(self):
        """Verify index subcommand can request a full vector-store reset."""

        args = parse_args(["index", "--reset"])
        commands = build_stage_commands(args)

        self.assertEqual(commands, [[sys.executable, str(Path("scripts") / "build_index.py"), "--reset"]])

    def test_build_index_collects_only_final_chunk_files(self):
        """Verify legacy *_chunks_preview.json files are not collected for indexing."""

        class Args:
            chunks_path = None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_path = root / "paper_chunks.json"
            preview_path = root / "paper_chunks_preview.json"
            final_path.write_text("[]", encoding="utf-8")
            preview_path.write_text("[]", encoding="utf-8")
            args = Args()
            args.chunks_dir = root

            paths = _collect_chunk_paths(args)

        self.assertEqual(paths, [final_path.resolve()])


if __name__ == "__main__":
    unittest.main()
