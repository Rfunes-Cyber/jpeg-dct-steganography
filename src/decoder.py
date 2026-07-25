# CS 4463 - Steganography
# Group 6 - Milestone #2
#
# Team Members:
# Raymond Funes
# Jermaine K Morehead
# Shruthi Senthilarasu
#
# JPEG DCT Coefficient Parity Steganography

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from encoder import (
    HEADER_BITS,
    coefficient_parity,
    count_selected_coefficients,
    get_embedding_preset,
    iter_selected_coefficients,
)
from jpeg_dct_handler import JPEGDCTHandler


@dataclass(frozen=True)
class DecodingResult:
    """Summary of a decoding operation."""

    preset_name: str
    max_bits_per_block: int | None
    header_bits: int
    payload_bits: int
    message_bytes: int
    output_path: str


def _read_bits(
    coefficient_iterator: Iterator[tuple[object, int]],
    bit_count: int,
    context: str,
) -> str:
    """Pull `bit_count` parity bits from the coefficient stream.

    The encoder writes one message bit per selected coefficient by setting
    that coefficient's parity, in the exact order produced by
    `iter_selected_coefficients()` for the chosen preset. Reusing the same
    iterator instance across both the header read and the payload read
    (rather than restarting it) is what keeps this function aligned with
    where the encoder left off.
    """

    bits = []

    for _ in range(bit_count):
        try:
            _, value = next(coefficient_iterator)
        except StopIteration as error:
            raise ValueError(
                f"Ran out of selected coefficients while reading {context}. "
                "The stego JPEG may be truncated, corrupted, embedded with "
                "a different preset than the one requested, or may not "
                "contain a hidden message at all."
            ) from error

        # Parity, not the coefficient's sign, carries the bit: abs(value) % 2
        # is identical for a coefficient and its negation, so positive and
        # negative coefficients are read the same way the encoder wrote them.
        bits.append(str(coefficient_parity(value)))

    return "".join(bits)


def _bits_to_bytes(bits: str) -> bytes:
    """Regroup a bit string into the original message bytes.

    The encoder only ever embeds whole ASCII bytes (see
    `prepare_payload` in encoder.py, which truncates to a multiple of 8),
    so it is safe to assume `bits` splits evenly into 8-bit groups here.
    """

    return bytes(
        int(bits[index:index + 8], 2)
        for index in range(0, len(bits), 8)
    )


def decode_message(
    input_path: str,
    output_path: str,
    preset_choice: int = 1,
) -> DecodingResult:
    """Extract a hidden message from a stego JPEG and write it to disk.

    `preset_choice` must match the preset used when the message was
    encoded (see encoder.PRESETS): the embedding preset determines which
    coefficients the encoder wrote bits into (the "Balanced" and "Low
    Detectability" presets skip coefficients past a per-block limit), and
    that choice is not recorded anywhere in the stego JPEG itself. Getting
    it wrong reads the wrong bits, not necessarily a clean error, so the
    preset must be supplied out of band (e.g. agreed on with the sender)
    just as it must be supplied to encode_message().
    """

    # Raises the same ValueError as encode_message() for an invalid
    # preset number, so callers see one consistent error for this input.
    preset_name, max_bits_per_block = get_embedding_preset(preset_choice)

    handler = JPEGDCTHandler()
    # Reuses the existing handler's load(): identical validation and
    # identical coefficient layout as what the encoder wrote against.
    handler.load(input_path)

    # Counting first (via its own fresh generator) tells us the total
    # embedding capacity so the header value can be sanity-checked before
    # we trust it, without disturbing the iterator used below.
    total_selected_coefficients = count_selected_coefficients(
        handler, max_bits_per_block,
    )

    if total_selected_coefficients < HEADER_BITS:
        raise ValueError(
            "This JPEG does not contain enough selected coefficients to "
            f"hold a {HEADER_BITS}-bit header using the {preset_name} "
            "preset. It cannot be a valid stego image produced by this "
            "encoder with this preset."
        )

    max_payload_bits = total_selected_coefficients - HEADER_BITS

    # One iterator, consumed first for the header and then for the payload,
    # so the payload starts exactly where the header left off -- matching
    # the encoder's single pass over `iter_selected_coefficients()` for
    # this same preset.
    coefficient_iterator = iter_selected_coefficients(
        handler, max_bits_per_block,
    )

    header_bits = _read_bits(
        coefficient_iterator,
        HEADER_BITS,
        context="the message-length header",
    )
    payload_bit_count = int(header_bits, 2)

    # Plausibility check: a genuine header can never claim more bits than
    # the image can hold, and the encoder only ever embeds whole bytes.
    # A JPEG with no real hidden message yields effectively random parity
    # bits here, so this check is what catches "no valid header" cases
    # instead of silently reading garbage as a message.
    if payload_bit_count < 0 or payload_bit_count > max_payload_bits:
        raise ValueError(
            f"Extracted message length ({payload_bit_count} bits) exceeds "
            f"the available capacity ({max_payload_bits} bits). This JPEG "
            "does not contain a valid hidden message for this encoder."
        )

    if payload_bit_count % 8 != 0:
        raise ValueError(
            f"Extracted message length ({payload_bit_count} bits) is not "
            "a whole number of bytes. This JPEG does not contain a valid "
            "hidden message for this encoder."
        )

    payload_bits = _read_bits(
        coefficient_iterator,
        payload_bit_count,
        context="the hidden message payload",
    )

    message_bytes = _bits_to_bytes(payload_bits)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Binary mode: the recovered bytes are not assumed to be text, so
        # they must round-trip exactly regardless of what they represent.
        with open(path, "wb") as output_file:
            output_file.write(message_bytes)
    except OSError as error:
        raise RuntimeError(
            f"Could not write decoded message to '{path}'."
        ) from error

    return DecodingResult(
        preset_name=preset_name,
        max_bits_per_block=max_bits_per_block,
        header_bits=HEADER_BITS,
        payload_bits=payload_bit_count,
        message_bytes=len(message_bytes),
        output_path=str(path),
    )


def print_decoding_result(result: DecodingResult) -> None:
    """Print decoding statistics without interpreting the recovered bytes.

    The recovered file may not be text, so its contents are never decoded
    or printed here -- only file-level statistics are reported. Anyone who
    wants to read the output as ASCII can open the file themselves.
    """

    if result.max_bits_per_block is None:
        block_limit = "Unlimited"
    else:
        block_limit = str(result.max_bits_per_block)

    print("\nDecoding complete.")
    print("Embedding preset:", result.preset_name)
    print("Maximum bits per block:", block_limit)
    print("Header bits read:", result.header_bits)
    print("Payload bits extracted:", result.payload_bits)
    print("Recovered message size:", f"{result.message_bytes} bytes")
    print("Written to:", result.output_path)


if __name__ == "__main__":
    from encoder import choose_embedding_preset

    stego_jpeg = input("Enter the stego JPEG path: ").strip()
    output_file_path = input("Enter the output file path: ").strip()

    try:
        selected_preset = choose_embedding_preset()

        decoding_result = decode_message(
            stego_jpeg,
            output_file_path,
            selected_preset,
        )

        print_decoding_result(decoding_result)

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"\nDecoding failed: {error}")
