from jpeg_dct_handler import JPEGDCTHandler


def main() -> None:
    handler = JPEGDCTHandler()

    handler.load("test.jpg")
    handler.print_info()
    handler.print_first_blocks()

    #handler.save("test_copy.jpg")

    print("\nSuccessfully created test_copy.jpg")


if __name__ == "__main__":
    main()