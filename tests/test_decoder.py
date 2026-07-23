"""
Tests for src/decoder.py.

These tests exercise the decoder against the *existing* encoder and DCT
handler exactly as they are written today:

* There is no zig-zag traversal in this codebase -- coefficients are
  visited in raster order within each block (see
  JPEGDCTHandler.iter_ac_coefficients), skipping the DC term.
* There is no "embedding level" / bits-per-block concept -- every AC
  coefficient with abs(value) > 1 carries exactly one bit
  (JPEGDCTHandler.is_usable / iter_usable_coefficients).

A dedicated test (test_no_embedding_level_parameter_exists) documents that
absence so a future change to the encoder's public API doesn't silently
break the assumptions these tests rely on.
"""

from pathlib import Path

import pytest

from decoder import decode_message, _read_bits
from encoder import (
    HEADER_BITS,
    change_parity,
    encode_message,
    integer_to_bits,
)
from jpeg_dct_handler import JPEGDCTHandler

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_JPEG = REPO_ROOT / "test.jpg"


def _embed_bits(input_path: Path, output_path: Path, bits: str) -> None:
    """Embed a raw bit string using the encoder's own primitives.

    This mirrors the inner loop of encoder.encode_message() exactly (same
    handler, same iter_usable_coefficients() order, same change_parity()),
    so tests can construct headers/payloads that encode_message's ASCII-only
    text_to_bits() would refuse (e.g. a header claiming an impossible
    length, or raw non-ASCII byte values).
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


def _bytes_to_bits(data: bytes) -> str:
    return "".join(format(byte, "08b") for byte in data)


# ---------------------------------------------------------------------------
# Round-trip tests: encode with the real encoder, decode with the new decoder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        pytest.param("Hi", id="short_text"),
        pytest.param("A", id="single_character"),
        pytest.param("", id="empty_message"),
    ],
)
def test_short_text(tmp_path, message):
    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.txt"

    encode_message(str(ORIGINAL_JPEG), str(stego_path), message)
    result = decode_message(str(stego_path), str(output_path))

    assert output_path.read_bytes() == message.encode("ascii")
    assert result.message_bytes == len(message)


def test_multiline_text(tmp_path):
    message = "Line one\nLine two\nLine three\n\nParagraph after blank line."
    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.txt"

    encode_message(str(ORIGINAL_JPEG), str(stego_path), message)
    decode_message(str(stego_path), str(output_path))

    assert output_path.read_bytes() == message.encode("ascii")


def test_punctuation(tmp_path):
    message = "Wait... what?! #1 @home, 100% (yes) [ok] {sure} <ok> \"quote\" 'apostrophe'"
    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.txt"

    encode_message(str(ORIGINAL_JPEG), str(stego_path), message)
    decode_message(str(stego_path), str(output_path))

    assert output_path.read_bytes() == message.encode("ascii")


def test_binary_data(tmp_path):
    """The decoder must be binary-safe even though encode_message() itself
    is ASCII-only (text_to_bits enforces this). Embedding raw bytes
    directly through the same primitives the encoder uses confirms the
    decoder reconstructs arbitrary byte values -- including 0x00 and
    0xFF -- rather than assuming printable ASCII."""

    raw_message = bytes(range(256))
    payload_bits = _bytes_to_bits(raw_message)
    header_bits = integer_to_bits(len(payload_bits), HEADER_BITS)

    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.bin"

    _embed_bits(ORIGINAL_JPEG, stego_path, header_bits + payload_bits)
    result = decode_message(str(stego_path), str(output_path))

    assert output_path.read_bytes() == raw_message
    assert result.message_bytes == len(raw_message)


def test_no_embedding_level_parameter_exists():
    """Documents that this codebase has no embedding-level / bits-per-block
    concept. If one is ever added to the encoder, this test should be
    replaced with real per-level coverage instead of silently passing."""

    import inspect

    assert "level" not in inspect.signature(encode_message).parameters
    assert "level" not in inspect.signature(decode_message).parameters


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------


def test_invalid_jpeg_raises(tmp_path):
    fake_jpeg = tmp_path / "not_really_a_jpeg.jpg"
    fake_jpeg.write_bytes(b"this is not JPEG data at all")

    with pytest.raises(ValueError):
        decode_message(str(fake_jpeg), str(tmp_path / "out.txt"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        decode_message(
            str(tmp_path / "does_not_exist.jpg"),
            str(tmp_path / "out.txt"),
        )


def test_no_valid_header_raises(tmp_path):
    """An unmodified JPEG's coefficient parities are not a message header,
    so the extracted length should fail the plausibility check (either by
    exceeding capacity or not being a whole number of bytes)."""

    with pytest.raises(ValueError):
        decode_message(str(ORIGINAL_JPEG), str(tmp_path / "out.txt"))


def test_impossible_length_raises(tmp_path):
    """A header claiming far more bits than the image could ever hold
    must be rejected before any payload bits are read."""

    huge_length = 2 ** 31
    header_bits = integer_to_bits(huge_length, HEADER_BITS)

    stego_path = tmp_path / "stego.jpg"
    _embed_bits(ORIGINAL_JPEG, stego_path, header_bits)

    with pytest.raises(ValueError, match="exceeds"):
        decode_message(str(stego_path), str(tmp_path / "out.txt"))


def test_truncated_message_raises():
    """Simulates running out of usable coefficients mid-payload by handing
    _read_bits an iterator with fewer items than the header promises --
    this is exactly the condition that a corrupted or truncated stego
    JPEG would trigger during payload extraction."""

    short_iterator = iter([(None, 3), (None, 5)])

    with pytest.raises(ValueError, match="Ran out of usable coefficients"):
        _read_bits(short_iterator, bit_count=5, context="the hidden message payload")
