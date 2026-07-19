from pathlib import Path

import jpeglib


class JPEGDCTHandler:
    """
    Reads, inspects, modifies, and writes JPEG DCT coefficients.
    """

    def __init__(self) -> None:
        self.jpeg = None
        self.input_path: Path | None = None

    def load(self, file_path: str) -> None:
        """
        Load quantized DCT coefficients from a JPEG file.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"JPEG file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        try:
            self.jpeg = jpeglib.read_dct(str(path))
        except Exception as error:
            raise ValueError(
                f"Could not read '{path}' as a JPEG file"
            ) from error

        self.input_path = path

    def print_info(self) -> None:
        """
        Print information about the loaded JPEG file.
        """
        self._require_jpeg_loaded()

        print(f"JPEG File: {self.input_path}")
        print(
            "Y (Luminance) Coefficients Shape:",
            self.jpeg.Y.shape,
        )
        print(f"Y coefficient data type: {self.jpeg.Y.dtype}")

        if self.jpeg.Cb is not None:
            print(
                "Cb (Blue-difference Chrominance) "
                f"Coefficients Shape: {self.jpeg.Cb.shape}"
            )
            print(f"Cb coefficient data type: {self.jpeg.Cb.dtype}")
        else:
            print(
                "Cb (Blue-difference Chrominance) "
                "Coefficients: Not present"
            )

        if self.jpeg.Cr is not None:
            print(
                "Cr (Red-difference Chrominance) "
                f"Coefficients Shape: {self.jpeg.Cr.shape}"
            )
            print(f"Cr coefficient data type: {self.jpeg.Cr.dtype}")
        else:
            print(
                "Cr (Red-difference Chrominance) "
                "Coefficients: Not present"
            )

        print(f"Quantization-table shape: {self.jpeg.qt.shape}")
        print("\nQuantization tables:")
        print(self.jpeg.qt)

    def print_first_blocks(self) -> None:
        """
        Print the first 8x8 DCT block from each component.
        """
        self._require_jpeg_loaded()

        for component_name, component in self.get_components():
            print(f"\nFirst 8x8 {component_name} DCT block:")
            print(component[0, 0])

    def save(self, output_path: str) -> None:
        """
        Save the modified DCT coefficients back to a JPEG file.
        """
        self._require_jpeg_loaded()

        try:
            self.jpeg.write_dct(str(output_path))
            print(f"Modified JPEG saved to: {output_path}")
        except Exception as error:
            raise RuntimeError(
                f"Could not write modified JPEG to '{output_path}'"
            ) from error

    @staticmethod
    def is_usable(value: int) -> bool:
        return abs(value) > 1

    def get_components(self):
        """
        Yield component names and their DCT coefficient arrays.
        """
        self._require_jpeg_loaded()

        yield "Y", self.jpeg.Y

        if self.jpeg.Cb is not None:
            yield "Cb", self.jpeg.Cb

        if self.jpeg.Cr is not None:
            yield "Cr", self.jpeg.Cr

    def _require_jpeg_loaded(self) -> None:
        """
        Ensure a JPEG has been loaded before performing operations.
        """
        if self.jpeg is None:
            raise ValueError(
                "No JPEG file loaded. Please load a JPEG file first."
            )