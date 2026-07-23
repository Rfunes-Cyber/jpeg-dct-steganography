"""Tests for src/cli.py: the unified argparse entry point over
encode_message()/decode_message()."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PATH = REPO_ROOT / "src" / "cli.py"
ORIGINAL_JPEG = REPO_ROOT / "test.jpg"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        capture_output=True,
        text=True,
    )


def test_encode_then_decode_round_trip(tmp_path):
    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.txt"
    message = "CLI round trip test"

    encode_result = _run_cli(
        "encode", str(ORIGINAL_JPEG), str(stego_path), message,
    )
    assert encode_result.returncode == 0, encode_result.stderr

    decode_result = _run_cli("decode", str(stego_path), str(output_path))
    assert decode_result.returncode == 0, decode_result.stderr

    assert output_path.read_bytes() == message.encode("ascii")


def test_decode_non_stego_jpeg_exits_nonzero(tmp_path):
    result = _run_cli(
        "decode", str(ORIGINAL_JPEG), str(tmp_path / "out.txt"),
    )

    assert result.returncode == 1
    assert "failed" in result.stderr.lower()


def test_missing_command_exits_nonzero():
    result = _run_cli()

    assert result.returncode != 0
