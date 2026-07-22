import argparse
import jpeglib
import numpy as np

from jpeg_dct_handler import JPEGDCTHandler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the JPEG DCT handler."
    )

    parser.add_argument(
        "input_jpeg",
        help="Path to the JPEG file to load.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="test_copy.jpg",
        help="Output JPEG path. Default: test_copy.jpg",
    )

    args = parser.parse_args()

    handler = JPEGDCTHandler()

    # TEST: Load the JPEG selected through the command line
    handler.load(args.input_jpeg)
    print("Load test passed.")

    # TEST: Print basic information
    handler.print_info()

    # TEST: Count usable coefficients
    usable_count = handler.count_usable_coefficients()
    print(f"\nUsable coefficients: {usable_count}")

    # TEST: Read the first usable coefficient
    location, value = next(handler.iter_usable_coefficients())
    print(f"First usable coefficient: {location} = {value}")

    # TEST: Confirm get_coefficient() returns the same value
    print(
        "Coefficient read test:",
        handler.get_coefficient(location) == value,
    )

    # TEST: Save an unchanged copy
    handler.save(args.output)
    print("Save test passed.")

    # TEST: Verify that coefficients survived saving
    original = jpeglib.read_dct(args.input_jpeg)
    copied = jpeglib.read_dct(args.output)

    print("\nRound-trip comparison:")
    print("Y matches:", np.array_equal(original.Y, copied.Y))
    print("Cb matches:", np.array_equal(original.Cb, copied.Cb))
    print("Cr matches:", np.array_equal(original.Cr, copied.Cr))
    print(
        "Quantization tables match:",
        np.array_equal(original.qt, copied.qt),
    )


if __name__ == "__main__":
    main()