import argparse
import sys
from pathlib import Path

from decoder import decode_message, print_decoding_result
from encoder import encode_message, print_encoding_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hide or extract a secret message in a JPEG's quantized "
            "DCT coefficients using coefficient-parity steganography."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser(
        "encode",
        help="Embed a message into a JPEG.",
    )
    encode_parser.add_argument("input_jpeg", help="Path to the cover JPEG.")
    encode_parser.add_argument("output_jpeg", help="Path to write the stego JPEG.")
    encode_parser.add_argument("message", help="Secret message to embed (ASCII only).")

    decode_parser = subparsers.add_parser(
        "decode",
        help="Extract a message from a stego JPEG.",
    )
    decode_parser.add_argument("input_jpeg", help="Path to the stego JPEG.")
    decode_parser.add_argument(
        "output_file",
        help="Path to write the recovered message (raw bytes).",
    )
    decode_parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Also print the recovered message as ASCII text. Off by "
            "default because the underlying decode is binary-safe and "
            "should not assume recovered bytes are text."
        ),
    )

    return parser


def _display_as_ascii(output_path: str) -> None:
    """Print the recovered file's contents as ASCII text.

    Only called when the user explicitly passes --show: decode_message()
    itself stays binary-safe, so this reads the bytes back from disk
    rather than having the decoder assume text.
    """

    recovered_bytes = Path(output_path).read_bytes()

    try:
        message = recovered_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(
            "Recovered data is not ASCII text -- inspect the output "
            f"file directly: {output_path}"
        ) from error

    print("\nSecret message:")
    print(message)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "encode":
            result = encode_message(
                args.input_jpeg,
                args.output_jpeg,
                args.message,
            )
            print_encoding_result(result)
        else:
            result = decode_message(
                args.input_jpeg,
                args.output_file,
            )
            print_decoding_result(result)

            if args.show:
                _display_as_ascii(args.output_file)

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\n{args.command.capitalize()} failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
