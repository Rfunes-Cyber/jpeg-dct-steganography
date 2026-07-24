from dataclasses import dataclass

from jpeg_dct_handler import JPEGDCTHandler


HEADER_BITS = 32

PRESETS = {
    1: ("Maximum Capacity", None),
    2: ("Balanced", 4),
    3: ("Low Detectability", 2),
}


@dataclass(frozen=True)
class EncodingResult:
    """Summary of an encoding operation."""

    preset_name: str
    max_bits_per_block: int | None
    original_message_bytes: int
    embedded_message_bytes: int
    embedded_payload_bits: int
    modified_coefficients: int
    available_payload_bytes: int
    message_was_truncated: bool


def choose_embedding_preset() -> int:
    """Prompt the user to choose an embedding preset."""

    print("\nEmbedding Preset")
    print()
    print("1 - Maximum Capacity")
    print("    Unlimited bits/block")
    print()
    print("2 - Balanced")
    print("    4 bits/block")
    print()
    print("3 - Low Detectability")
    print("    2 bits/block")

    while True:
        choice = input("\nSelect a preset (1-3): ").strip()

        if choice in {"1", "2", "3"}:
            return int(choice)

        print("Invalid selection. Please choose 1, 2, or 3.")


def get_embedding_preset(
    preset_choice: int,
) -> tuple[str, int | None]:
    """Return the name and block limit for a preset."""

    if preset_choice not in PRESETS:
        raise ValueError(
            "The embedding preset must be 1, 2, or 3."
        )

    return PRESETS[preset_choice]


def text_to_bits(text: str) -> str:
    """Convert ASCII text into an 8-bit binary string."""

    try:
        encoded_text = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(
            "The secret message must contain ASCII characters only."
        ) from error

    return "".join(
        format(byte, "08b")
        for byte in encoded_text
    )


def integer_to_bits(value: int, bit_count: int) -> str:
    """Convert an integer into a fixed-length binary string."""

    if value < 0:
        raise ValueError("The integer cannot be negative.")

    maximum_value = (1 << bit_count) - 1

    if value > maximum_value:
        raise ValueError(
            f"The value {value} cannot fit in {bit_count} bits."
        )

    return format(value, f"0{bit_count}b")


def coefficient_parity(value: int) -> int:
    """Return the parity represented by a DCT coefficient."""

    return abs(value) % 2


def change_parity(value: int, required_bit: int) -> int:
    """Modify a coefficient so its parity matches the required bit."""

    if required_bit not in {0, 1}:
        raise ValueError(
            "The required bit must be either 0 or 1."
        )

    if not JPEGDCTHandler.is_usable(value):
        raise ValueError(
            f"Coefficient {value} is not usable for embedding."
        )

    if coefficient_parity(value) == required_bit:
        return value

    if value > 0:
        return value + 1

    return value - 1


def iter_selected_coefficients(
    handler: JPEGDCTHandler,
    max_bits_per_block: int | None,
):
    """Yield coefficients allowed by the selected preset."""

    current_block = None
    bits_used_in_block = 0

    for location, value in handler.iter_usable_coefficients():
        block_key = (
            location.component,
            location.block_row,
            location.block_column,
        )

        if block_key != current_block:
            current_block = block_key
            bits_used_in_block = 0

        if (
            max_bits_per_block is not None
            and bits_used_in_block >= max_bits_per_block
        ):
            continue

        yield location, value
        bits_used_in_block += 1


def count_selected_coefficients(
    handler: JPEGDCTHandler,
    max_bits_per_block: int | None,
) -> int:
    """Count coefficients allowed by the selected preset."""

    return sum(
        1
        for _ in iter_selected_coefficients(
            handler,
            max_bits_per_block,
        )
    )


def prepare_payload(
    message: str,
    available_payload_bits: int,
) -> tuple[str, int, bool]:
    """Convert and truncate the message to fit the capacity."""

    message_bits = text_to_bits(message)
    original_message_bytes = len(message_bits) // 8

    usable_payload_bits = max(
        0,
        available_payload_bits,
    )

    # Only embed complete ASCII characters.
    usable_payload_bits -= usable_payload_bits % 8

    embedded_bits = message_bits[:usable_payload_bits]
    was_truncated = len(embedded_bits) < len(message_bits)

    return (
        embedded_bits,
        original_message_bytes,
        was_truncated,
    )


def encode_message(
    input_path: str,
    output_path: str,
    message: str,
    preset_choice: int = 1,
) -> EncodingResult:
    """Embed an ASCII message using the selected preset."""

    preset_name, max_bits_per_block = (
        get_embedding_preset(preset_choice)
    )

    handler = JPEGDCTHandler()
    handler.load(input_path)

    usable_coefficient_count = (
        count_selected_coefficients(
            handler,
            max_bits_per_block,
        )
    )

    if usable_coefficient_count < HEADER_BITS:
        raise ValueError(
            "This JPEG does not contain enough usable coefficients "
            f"for the {HEADER_BITS}-bit message header using the "
            f"{preset_name} preset."
        )

    available_payload_bits = (
        usable_coefficient_count - HEADER_BITS
    )

    available_payload_bytes = (
        available_payload_bits // 8
    )

    payload_bits, original_message_bytes, was_truncated = (
        prepare_payload(
            message,
            available_payload_bits,
        )
    )

    header = integer_to_bits(
        len(payload_bits),
        HEADER_BITS,
    )

    complete_bitstream = header + payload_bits
    modified_coefficients = 0

    coefficient_iterator = iter_selected_coefficients(
        handler,
        max_bits_per_block,
    )

    for bit_character in complete_bitstream:
        location, current_value = next(
            coefficient_iterator
        )

        required_bit = int(bit_character)

        new_value = change_parity(
            current_value,
            required_bit,
        )

        if new_value != current_value:
            handler.set_coefficient(
                location,
                new_value,
            )

            modified_coefficients += 1

    handler.save(output_path)

    return EncodingResult(
        preset_name=preset_name,
        max_bits_per_block=max_bits_per_block,
        original_message_bytes=original_message_bytes,
        embedded_message_bytes=len(payload_bits) // 8,
        embedded_payload_bits=len(payload_bits),
        modified_coefficients=modified_coefficients,
        available_payload_bytes=available_payload_bytes,
        message_was_truncated=was_truncated,
    )


def print_encoding_result(
    result: EncodingResult,
) -> None:
    """Print encoding statistics."""

    if result.max_bits_per_block is None:
        block_limit = "Unlimited"
    else:
        block_limit = str(result.max_bits_per_block)

    print("\nEncoding complete.")
    print("Embedding preset:", result.preset_name)
    print("Maximum bits per block:", block_limit)
    print(
        "Available message capacity:",
        f"{result.available_payload_bytes} bytes",
    )
    print(
        "Original message size:",
        f"{result.original_message_bytes} bytes",
    )
    print(
        "Embedded message size:",
        f"{result.embedded_message_bytes} bytes",
    )
    print(
        "Embedded payload bits:",
        result.embedded_payload_bits,
    )
    print(
        "Modified coefficients:",
        result.modified_coefficients,
    )

    if result.message_was_truncated:
        print(
            "Warning: The complete message did not fit. "
            "Only the maximum number of complete ASCII "
            "characters was embedded."
        )


if __name__ == "__main__":
    input_jpeg = input(
        "Enter the input JPEG path: "
    ).strip()

    output_jpeg = input(
        "Enter the output JPEG path: "
    ).strip()

    secret_message = input(
        "Enter the secret message: "
    )

    try:
        selected_preset = choose_embedding_preset()

        encoding_result = encode_message(
            input_jpeg,
            output_jpeg,
            secret_message,
            selected_preset,
        )

        print_encoding_result(encoding_result)

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"\nEncoding failed: {error}")