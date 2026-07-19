def text_to_bits(text: str) -> str:
    """
    Convert a text string into a binary string.
    Example:
        "Hi" -> "0100100001101001"
    """
    bit_string = ""

    for character in text:
        bit_string += format(ord(character), "08b")

    return bit_string


def bits_to_text(bits: str) -> str:
    """
    Convert a binary string back into text.
    Example:
        "0100100001101001" -> "Hi"
    """
    text = ""

    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        text += chr(int(byte, 2))

    return text


if __name__ == "__main__":
    message = "Hello"

    bits = text_to_bits(message)
    print("Binary:", bits)

    decoded = bits_to_text(bits)
    print("Decoded:", decoded)