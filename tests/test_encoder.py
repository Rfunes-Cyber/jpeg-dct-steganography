"""
Tests for src/encoder.py.

These exercise the encoder's helper functions directly (bit conversion,
parity, truncation) as well as encode_message() end to end. Decoder-side
round-trip correctness is already covered in tests/test_decoder.py; this
file focuses on the encoder's own contract and error handling.
"""

from pathlib import Path

import pytest

from encoder import (
    HEADER_BITS,
    change_parity,
    coefficient_parity,
    count_selected_coefficients,
    encode_message,
    get_embedding_preset,
    integer_to_bits,
    iter_selected_coefficients,
    prepare_payload,
    text_to_bits,
)
from jpeg_dct_handler import JPEGDCTHandler

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_JPEG = REPO_ROOT / "test.jpg"
TINY_SOLID_JPEG = REPO_ROOT / "tiny_solid.jpg"


# ---------------------------------------------------------------------------
# text_to_bits
# ---------------------------------------------------------------------------


def test_text_to_bits_converts_ascii():
    assert text_to_bits("A") == "01000001"
    assert text_to_bits("Hi") == "0100100001101001"


def test_text_to_bits_empty_string_yields_empty_bits():
    assert text_to_bits("") == ""


def test_text_to_bits_rejects_non_ascii():
    with pytest.raises(ValueError):
        text_to_bits("café")


# ---------------------------------------------------------------------------
# integer_to_bits
# ---------------------------------------------------------------------------


def test_integer_to_bits_fixed_width():
    assert integer_to_bits(5, 8) == "00000101"
    assert integer_to_bits(0, 8) == "00000000"
    assert integer_to_bits(255, 8) == "11111111"


def test_integer_to_bits_rejects_negative():
    with pytest.raises(ValueError):
        integer_to_bits(-1, 8)


def test_integer_to_bits_rejects_overflow():
    with pytest.raises(ValueError):
        integer_to_bits(256, 8)


# ---------------------------------------------------------------------------
# coefficient_parity / change_parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected_parity",
    [
        (2, 0),
        (3, 1),
        (-2, 0),
        (-3, 1),
        (4, 0),
        (-4, 0),
    ],
)
def test_coefficient_parity_uses_magnitude(value, expected_parity):
    """Parity must depend on magnitude, not sign, so positive and
    negative coefficients of the same magnitude read identically."""

    assert coefficient_parity(value) == expected_parity


def test_change_parity_leaves_matching_value_unchanged():
    assert change_parity(4, required_bit=0) == 4
    assert change_parity(3, required_bit=1) == 3


@pytest.mark.parametrize(
    "value, required_bit, expected",
    [
        (4, 1, 5),
        (3, 0, 4),
        (-4, 1, -5),
        (-3, 0, -4),
    ],
)
def test_change_parity_adjusts_mismatched_value(value, required_bit, expected):
    """Positive coefficients are nudged up, negative coefficients are
    nudged down (further from zero) -- both preserve sign and land on
    the requested parity."""

    result = change_parity(value, required_bit)

    assert result == expected
    assert coefficient_parity(result) == required_bit


def test_change_parity_rejects_unusable_coefficient():
    for unusable_value in (-1, 0, 1):
        with pytest.raises(ValueError):
            change_parity(unusable_value, required_bit=0)


def test_change_parity_rejects_invalid_bit():
    with pytest.raises(ValueError):
        change_parity(4, required_bit=2)


# ---------------------------------------------------------------------------
# prepare_payload
# ---------------------------------------------------------------------------


def test_prepare_payload_fits_without_truncation():
    bits, original_bytes, was_truncated = prepare_payload(
        "hello", available_payload_bits=1000,
    )

    assert bits == text_to_bits("hello")
    assert original_bytes == 5
    assert was_truncated is False


def test_prepare_payload_truncates_to_fit():
    message = "hello world"
    message_bit_count = len(text_to_bits(message))

    bits, original_bytes, was_truncated = prepare_payload(
        message, available_payload_bits=message_bit_count - 8,
    )

    assert bits == text_to_bits(message)[: message_bit_count - 8]
    assert original_bytes == len(message)
    assert was_truncated is True


def test_prepare_payload_only_keeps_whole_bytes():
    """available_payload_bits that isn't a multiple of 8 must not produce
    a partial trailing character."""

    message = "hello"

    bits, _, was_truncated = prepare_payload(
        message, available_payload_bits=20,
    )

    assert len(bits) == 16
    assert was_truncated is True


def test_prepare_payload_zero_capacity_yields_empty_payload():
    bits, original_bytes, was_truncated = prepare_payload(
        "hello", available_payload_bits=0,
    )

    assert bits == ""
    assert original_bytes == 5
    assert was_truncated is True


# ---------------------------------------------------------------------------
# encode_message end to end
# ---------------------------------------------------------------------------


def test_encode_message_reports_accurate_stats(tmp_path):
    message = "stats check"
    stego_path = tmp_path / "stego.jpg"

    handler = JPEGDCTHandler()
    handler.load(str(ORIGINAL_JPEG))
    expected_capacity = handler.payload_capacity_bytes(HEADER_BITS)

    result = encode_message(str(ORIGINAL_JPEG), str(stego_path), message)

    assert result.original_message_bytes == len(message)
    assert result.embedded_message_bytes == len(message)
    assert result.embedded_payload_bits == len(message) * 8
    assert result.message_was_truncated is False
    assert result.available_payload_bytes == expected_capacity
    assert stego_path.exists()


def test_encode_message_rejects_non_ascii(tmp_path):
    with pytest.raises(ValueError):
        encode_message(str(ORIGINAL_JPEG), str(tmp_path / "stego.jpg"), "café")


def test_encode_message_rejects_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        encode_message(
            str(tmp_path / "does_not_exist.jpg"),
            str(tmp_path / "stego.jpg"),
            "hello",
        )


def test_encode_message_rejects_invalid_jpeg(tmp_path):
    fake_jpeg = tmp_path / "not_a_jpeg.jpg"
    fake_jpeg.write_bytes(b"not a real jpeg")

    with pytest.raises(ValueError):
        encode_message(str(fake_jpeg), str(tmp_path / "stego.jpg"), "hello")


def test_encode_message_rejects_invalid_output_extension(tmp_path):
    with pytest.raises(ValueError):
        encode_message(str(ORIGINAL_JPEG), str(tmp_path / "stego.png"), "hello")


def test_encode_message_rejects_jpeg_with_no_usable_capacity(tmp_path):
    """A solid-color JPEG has essentially no non-zero, non-unit AC
    coefficients, so it cannot even hold the fixed-length header."""

    with pytest.raises(ValueError, match="enough selected coefficients"):
        encode_message(
            str(TINY_SOLID_JPEG), str(tmp_path / "stego.jpg"), "hi",
        )


# ---------------------------------------------------------------------------
# Embedding presets
# ---------------------------------------------------------------------------


def test_get_embedding_preset_returns_expected_presets():
    assert get_embedding_preset(1) == ("Maximum Capacity", None)
    assert get_embedding_preset(2) == ("Balanced", 4)
    assert get_embedding_preset(3) == ("Low Detectability", 2)


def test_get_embedding_preset_rejects_invalid_choice():
    with pytest.raises(ValueError, match="1, 2, or 3"):
        get_embedding_preset(4)

    with pytest.raises(ValueError, match="1, 2, or 3"):
        get_embedding_preset(0)


@pytest.mark.parametrize("max_bits_per_block", [1, 2, 4])
def test_iter_selected_coefficients_never_exceeds_per_block_limit(max_bits_per_block):
    """The per-block cap is the whole point of the "Balanced" / "Low
    Detectability" presets: verify it directly against the real handler
    rather than only indirectly through a full encode/decode round trip."""

    handler = JPEGDCTHandler()
    handler.load(str(ORIGINAL_JPEG))

    bits_used_per_block: dict[tuple[str, int, int], int] = {}

    for location, _ in iter_selected_coefficients(handler, max_bits_per_block):
        block_key = (location.component, location.block_row, location.block_column)
        bits_used_per_block[block_key] = bits_used_per_block.get(block_key, 0) + 1

    assert bits_used_per_block
    assert max(bits_used_per_block.values()) <= max_bits_per_block


def test_lower_bits_per_block_never_increases_capacity():
    """A stricter per-block cap should select fewer (or equal) coefficients
    overall than a looser one -- capacity should shrink monotonically as
    detectability improves (Low Detectability <= Balanced <= Maximum)."""

    handler = JPEGDCTHandler()
    handler.load(str(ORIGINAL_JPEG))

    unlimited_count = count_selected_coefficients(handler, None)
    balanced_count = count_selected_coefficients(handler, 4)
    low_detectability_count = count_selected_coefficients(handler, 2)

    assert low_detectability_count <= balanced_count <= unlimited_count


def test_encode_message_reports_selected_preset(tmp_path):
    result = encode_message(
        str(ORIGINAL_JPEG), str(tmp_path / "stego.jpg"), "hi", 2,
    )

    assert result.preset_name == "Balanced"
    assert result.max_bits_per_block == 4


def test_encode_message_rejects_invalid_preset_choice(tmp_path):
    with pytest.raises(ValueError, match="1, 2, or 3"):
        encode_message(
            str(ORIGINAL_JPEG), str(tmp_path / "stego.jpg"), "hi", 5,
        )
