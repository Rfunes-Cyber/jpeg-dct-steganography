"""
Tests for src/decoder.py.

These tests exercise the decoder against the *existing* encoder and DCT
handler exactly as they are written today:

* There is no zig-zag traversal in this codebase -- coefficients are
  visited in raster order within each block (see
  JPEGDCTHandler.iter_ac_coefficients), skipping the DC term.
* The encoder supports three embedding presets (encoder.PRESETS) that
  cap how many coefficients get used per 8x8 block: "Maximum Capacity"
  (unlimited), "Balanced" (4 per block), and "Low Detectability" (2 per
  block). The preset is *not* recorded in the stego JPEG itself, so
  decode_message() must be told which preset was used, exactly like
  encode_message() must be told which one to use.

Unless a test explicitly passes a preset_choice, both encode_message()
and decode_message() default to preset 1 (Maximum Capacity, no per-block
limit), which behaves identically to the original unfiltered
iter_usable_coefficients() traversal -- so most tests below are
unaffected by the presets' existence.
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
GRAYSCALE_JPEG = REPO_ROOT / "grayscale_test.jpg"


def _payload_capacity_bytes(jpeg_path: Path) -> int:
    """Report the same capacity encode_message() itself relies on, so
    boundary tests stay correct even if the fixture JPEG changes."""

    handler = JPEGDCTHandler()
    handler.load(str(jpeg_path))
    return handler.payload_capacity_bytes(HEADER_BITS)


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


def test_message_at_exact_capacity_boundary(tmp_path):
    """A message that uses every last available payload bit is the
    off-by-one risk case: one bit short or over would either lose data
    or run past the image's usable coefficients."""

    capacity_bytes = _payload_capacity_bytes(ORIGINAL_JPEG)
    message = "A" * capacity_bytes

    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.txt"

    encoding_result = encode_message(str(ORIGINAL_JPEG), str(stego_path), message)
    assert encoding_result.message_was_truncated is False

    decoding_result = decode_message(str(stego_path), str(output_path))

    assert output_path.read_bytes() == message.encode("ascii")
    assert decoding_result.message_bytes == capacity_bytes


def test_decodes_truncated_message_correctly(tmp_path):
    """When a message is too big to fit, encode_message() truncates it to
    whole ASCII characters before embedding. The decoder should recover
    exactly that truncated payload -- not the full original message and
    not garbage past the truncation point."""

    capacity_bytes = _payload_capacity_bytes(ORIGINAL_JPEG)
    oversized_message = "B" * (capacity_bytes + 500)

    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.txt"

    encoding_result = encode_message(
        str(ORIGINAL_JPEG), str(stego_path), oversized_message,
    )
    assert encoding_result.message_was_truncated is True

    decoding_result = decode_message(str(stego_path), str(output_path))

    expected = oversized_message[:capacity_bytes].encode("ascii")
    assert output_path.read_bytes() == expected
    assert decoding_result.message_bytes == capacity_bytes


def test_decode_is_read_only_and_idempotent(tmp_path):
    """Decoding must not mutate the stego JPEG, and decoding the same file
    twice must yield identical results -- a decoder that (accidentally)
    wrote back to the input, or depended on hidden iterator state, would
    fail this."""

    message = "idempotent decode check"
    stego_path = tmp_path / "stego.jpg"
    first_output = tmp_path / "first.txt"
    second_output = tmp_path / "second.txt"

    encode_message(str(ORIGINAL_JPEG), str(stego_path), message)
    stego_bytes_before = stego_path.read_bytes()

    decode_message(str(stego_path), str(first_output))
    decode_message(str(stego_path), str(second_output))

    assert first_output.read_bytes() == second_output.read_bytes() == message.encode("ascii")
    assert stego_path.read_bytes() == stego_bytes_before


def test_decode_creates_missing_output_directories(tmp_path):
    """The output path's parent directories may not exist yet; the decoder
    should create them rather than raising a bare FileNotFoundError."""

    message = "nested output path"
    stego_path = tmp_path / "stego.jpg"
    nested_output = tmp_path / "a" / "b" / "c" / "recovered.txt"

    encode_message(str(ORIGINAL_JPEG), str(stego_path), message)
    decode_message(str(stego_path), str(nested_output))

    assert nested_output.read_bytes() == message.encode("ascii")


def test_decode_grayscale_jpeg_with_no_chroma_components(tmp_path):
    """A grayscale JPEG has no Cb/Cr components at all
    (JPEGDCTHandler.get_components() only yields "Y" in that case). The
    decoder must work using only the Y component's usable coefficients."""

    message = "grayscale only"
    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.txt"

    encode_message(str(GRAYSCALE_JPEG), str(stego_path), message)
    decode_message(str(stego_path), str(output_path))

    assert output_path.read_bytes() == message.encode("ascii")


@pytest.mark.parametrize(
    "preset_choice",
    [
        pytest.param(1, id="preset_1_maximum_capacity"),
        pytest.param(2, id="preset_2_balanced"),
        pytest.param(3, id="preset_3_low_detectability"),
    ],
)
def test_each_embedding_preset_round_trips(tmp_path, preset_choice):
    """Every preset must encode and decode correctly as long as the same
    preset is used on both sides -- the presets only change which
    coefficients are selected, not the parity/header rules."""

    message = "Preset round trip check!"
    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.txt"

    encoding_result = encode_message(
        str(ORIGINAL_JPEG), str(stego_path), message, preset_choice,
    )
    decoding_result = decode_message(
        str(stego_path), str(output_path), preset_choice,
    )

    assert output_path.read_bytes() == message.encode("ascii")
    assert decoding_result.preset_name == encoding_result.preset_name
    assert decoding_result.max_bits_per_block == encoding_result.max_bits_per_block


def test_mismatched_preset_fails_to_decode(tmp_path):
    """The stego JPEG does not record which preset was used, so decoding
    with the wrong preset must not silently produce a plausible-looking
    but wrong message -- it should fail the header/plausibility checks
    because the coefficient sequence itself is different."""

    message = "this should not decode correctly"
    stego_path = tmp_path / "stego.jpg"
    output_path = tmp_path / "recovered.txt"

    encode_message(str(ORIGINAL_JPEG), str(stego_path), message, 1)

    with pytest.raises(ValueError):
        decode_message(str(stego_path), str(output_path), 3)


def test_decode_rejects_invalid_preset_choice(tmp_path):
    """decode_message() must validate its preset argument the same way
    encode_message() does, since both rely on encoder.get_embedding_preset()."""

    with pytest.raises(ValueError, match="1, 2, or 3"):
        decode_message(str(ORIGINAL_JPEG), str(tmp_path / "out.txt"), 4)


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

    with pytest.raises(ValueError, match="Ran out of selected coefficients"):
        _read_bits(short_iterator, bit_count=5, context="the hidden message payload")
