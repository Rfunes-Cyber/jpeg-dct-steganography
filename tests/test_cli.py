"""Tests for src/cli.py: the unified argparse entry point over
encode_message()/decode_message()."""

import subprocess
import sys
from pathlib import Path

from encoder import HEADER_BITS, change_parity, integer_to_bits
from jpeg_dct_handler import JPEGDCTHandler

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PATH = REPO_ROOT / "src" / "cli.py"
ORIGINAL_JPEG = REPO_ROOT / "test.jpg"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        capture_output=True,
        text=True,
    )


def _embed_bits(input_path: Path, output_path: Path, bits: str) -> None:
    """Embed a raw bit string using the encoder's own primitives.

    Mirrors encode_message()'s inner loop so tests can craft a stego
    JPEG containing bytes that encode_message()'s ASCII-only
    text_to_bits() would refuse (e.g. non-ASCII bytes for --show).
    """

    handler = JPEGDCTHandler()
    handler.load(str(input_path))

    coefficient_iterator = handler.iter_usable_coefficients()

    for bit_character in bits:
        location, current_value = next(coefficient_iterator)
        new_value = change_parity(current_value, int(bit_character))

        if new_value != current_value:
            handler.set_coefficient(location, new_value)

    handler.save(str(output_path))


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


def test_decode_show_flag_prints_message(tmp_path):
    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.txt"
    message = "hello from --show"

    encode_result = _run_cli(
        "encode", str(ORIGINAL_JPEG), str(stego_path), message,
    )
    assert encode_result.returncode == 0, encode_result.stderr

    decode_result = _run_cli(
        "decode", str(stego_path), str(output_path), "--show",
    )

    assert decode_result.returncode == 0, decode_result.stderr
    assert message in decode_result.stdout


def test_decode_show_flag_on_non_ascii_reports_clear_error(tmp_path):
    raw_message = bytes([0, 200, 255, 10, 65])
    payload_bits = "".join(format(byte, "08b") for byte in raw_message)
    header_bits = integer_to_bits(len(payload_bits), HEADER_BITS)

    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.bin"

    _embed_bits(ORIGINAL_JPEG, stego_path, header_bits + payload_bits)

    decode_result = _run_cli(
        "decode", str(stego_path), str(output_path), "--show",
    )

    assert decode_result.returncode == 1
    assert "not ascii" in decode_result.stderr.lower()
    # The decode itself must still have written the raw bytes to disk
    # even though --show's ASCII interpretation failed afterward.
    assert output_path.read_bytes() == raw_message
