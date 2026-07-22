from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:
    import jpeglib  # type: ignore[import]
except ImportError as error:
    raise ImportError(
        "The required dependency 'jpeglib' is missing. "
        "Install it with `pip install jpeglib`."
    ) from error

import numpy as np


@dataclass(frozen=True)
class CoefficientLocation:
    """Exact position of one DCT coefficient inside a JPEG."""

    component: str
    block_row: int
    block_column: int
    coefficient_row: int
    coefficient_column: int


class JPEGDCTHandler:
    """Loads, reads, changes, and saves quantized JPEG DCT coefficients."""

    def __init__(self) -> None:
        self.jpeg = None
        self.input_path: Path | None = None

    # -------------------------------------------------------------------------
    # FILE HANDLING
    # -------------------------------------------------------------------------

    def load(self, file_path: str) -> None:
        """Load a JPEG and its quantized DCT coefficients."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"JPEG file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError("Input file must be a .jpg or .jpeg file.")

        try:
            self.jpeg = jpeglib.read_dct(str(path))
        except Exception as error:
            raise ValueError(
                f"Could not read '{path}' as a JPEG file."
            ) from error

        self.input_path = path

    def save(self, output_path: str) -> None:
        """Save the current DCT coefficients as a JPEG."""

        self._require_loaded()
        path = Path(output_path)

        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError("Output file must be a .jpg or .jpeg file.")

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.jpeg.write_dct(str(path))
        except Exception as error:
            raise RuntimeError(
                f"Could not save JPEG to '{path}'."
            ) from error

    # -------------------------------------------------------------------------
    # INFORMATION / DEBUGGING
    # -------------------------------------------------------------------------

    def print_info(self) -> None:
        """Print component shapes, data types, and quantization tables."""

        self._require_loaded()
        print(f"JPEG: {self.input_path}")

        for name, component in self.get_components():
            print(f"{name} shape: {component.shape}")
            print(f"{name} type: {component.dtype}")

        print(f"Quantization table shape: {self.jpeg.qt.shape}")
        print(self.jpeg.qt)

    def print_first_blocks(self) -> None:
        """Print the first 8x8 block from each available component."""

        self._require_loaded()

        for name, component in self.get_components():
            print(f"\\nFirst {name} block:")
            print(component[0, 0])

    # -------------------------------------------------------------------------
    # COMPONENT ACCESS
    # -------------------------------------------------------------------------

    def get_components(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield JPEG components in Y, Cb, Cr order."""

        self._require_loaded()
        yield "Y", self.jpeg.Y

        if self.jpeg.Cb is not None:
            yield "Cb", self.jpeg.Cb

        if self.jpeg.Cr is not None:
            yield "Cr", self.jpeg.Cr

    def _get_component(self, name: str) -> np.ndarray:
        """Return one component array by name."""

        for component_name, component in self.get_components():
            if component_name == name:
                return component

        raise ValueError(f"JPEG does not contain component '{name}'.")

    # -------------------------------------------------------------------------
    # COEFFICIENT TRAVERSAL
    # -------------------------------------------------------------------------

    @staticmethod
    def is_usable(value: int) -> bool:
        """Skip -1, 0, and 1 because they are unsafe for embedding."""

        return abs(value) > 1

    def iter_ac_coefficients(
        self,
    ) -> Iterator[tuple[CoefficientLocation, int]]:
        """Yield every AC coefficient in a fixed order, skipping each block's DC."""

        self._require_loaded()

        for component_name, component in self.get_components():
            block_rows, block_columns, rows, columns = component.shape

            for block_row in range(block_rows):
                for block_column in range(block_columns):
                    for row in range(rows):
                        for column in range(columns):
                            if row == 0 and column == 0:
                                continue

                            location = CoefficientLocation(
                                component_name,
                                block_row,
                                block_column,
                                row,
                                column,
                            )

                            value = int(
                                component[
                                    block_row,
                                    block_column,
                                    row,
                                    column,
                                ]
                            )

                            yield location, value

    def iter_usable_coefficients(
        self,
    ) -> Iterator[tuple[CoefficientLocation, int]]:
        """Yield only AC coefficients with magnitude greater than one."""

        for location, value in self.iter_ac_coefficients():
            if self.is_usable(value):
                yield location, value

    # -------------------------------------------------------------------------
    # COEFFICIENT READING / WRITING
    # -------------------------------------------------------------------------

    def get_coefficient(self, location: CoefficientLocation) -> int:
        """Read one coefficient."""

        component = self._get_component(location.component)

        return int(
            component[
                location.block_row,
                location.block_column,
                location.coefficient_row,
                location.coefficient_column,
            ]
        )

    def set_coefficient(
        self,
        location: CoefficientLocation,
        new_value: int,
    ) -> None:
        """Change one coefficient in memory."""

        component = self._get_component(location.component)

        component[
            location.block_row,
            location.block_column,
            location.coefficient_row,
            location.coefficient_column,
        ] = new_value

    # -------------------------------------------------------------------------
    # CAPACITY
    # -------------------------------------------------------------------------

    def count_usable_coefficients(self) -> int:
        """Return raw embedding capacity in bits."""

        return sum(1 for _ in self.iter_usable_coefficients())

    def payload_capacity_bytes(self, header_bits: int = 32) -> int:
        """Return message capacity in bytes after reserving header bits."""

        if header_bits < 0:
            raise ValueError("Header bits cannot be negative.")

        usable_bits = self.count_usable_coefficients()
        return max(0, usable_bits - header_bits) // 8

    # -------------------------------------------------------------------------
    # INTERNAL CHECK
    # -------------------------------------------------------------------------

    def _require_loaded(self) -> None:
        """Prevent operations before load() succeeds."""

        if self.jpeg is None:
            raise RuntimeError("No JPEG has been loaded.")


