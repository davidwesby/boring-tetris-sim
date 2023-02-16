from __future__ import annotations
import fileinput

import numpy as np


class Grid:
    def __init__(self, height: int = 100, width: int = 10) -> None:
        self._state = np.zeros((height, width))

    def __eq__(self, other: Grid) -> bool:
        return np.array_equal(self._state, other._state)

    def add_piece(self, piece: Piece) -> None:
        """
        Drop `piece` into the grid and remove any rows it completes.
        """
        x = piece.column
        y = self._piece_resting_height(piece)
        self._state[y:y+piece.height, x:x+piece.width] = \
            np.logical_or(self._state[y:y+piece.height, x:x+piece.width], piece.pattern)
        self._remove_complete_rows()

    def highest_non_empty_row(self) -> int:
        """
        The height of the highest occupied cell in the entire grid.
        """
        peaks = self._get_peaks_in_columns(slice(0, len(self._state)))
        return np.max(peaks)

    def _get_peaks_in_columns(self, columns: slice) -> np.ndarray:
        """
        The heights of the highest occupied cell in each column in `columns`.
        """
        peaks = self._state.shape[0] - np.argmax(np.flipud(self._state[:, columns]), axis=0)
        empty_columns = np.logical_not(np.any(self._state[:, columns], axis=0))
        peaks[empty_columns] = 0
        return peaks

    def _piece_resting_height(self, piece: Piece) -> int:
        """
        The height of the bottom of the bounding box of `piece` when it is
        dropped into the grid.
        """
        peaks = self._get_peaks_in_columns(slice(piece.column, piece.column+piece.width))
        # Subtract height of "whitespace" within the piece.
        return np.max(peaks - piece.pattern.argmax(axis=0))

    def _remove_complete_rows(self) -> None:
        """
        Remove every row in which all the cells are occupied, and shift above
        rows down.
        """
        incomplete_rows = self._state[np.logical_not(np.all(self._state, axis=1))]
        n_complete_rows = self._state.shape[0] - incomplete_rows.shape[0]
        self._state = np.pad(incomplete_rows, ((0, n_complete_rows), (0, 0)))


class Piece:
    def __init__(self, pattern: np.ndarry, column: int) -> None:
        self.pattern = pattern
        self.height = pattern.shape[0]
        self.width = pattern.shape[1]
        self.column = column

    def __eq__(self, other: Piece) -> bool:
        return np.array_equal(self.pattern, other.pattern) and self.column == other.column

    @classmethod
    def from_specifier(cls, specifier: str) -> Piece:
        name = specifier[0]
        column = int(specifier[1:])
        match name:
            case 'I':
                pattern = np.array([[1, 1, 1, 1]])
            case 'J':
                pattern = np.array([[1, 1],
                                    [0, 1],
                                    [0, 1]])
            case 'L':
                pattern = np.array([[1, 1],
                                    [1, 0],
                                    [1, 0]])
            case 'Q':
                pattern = np.array([[1, 1],
                                    [1, 1]])
            case 'S':
                pattern = np.array([[1, 1, 0],
                                    [0, 1, 1]])
            case 'T':
                pattern = np.array([[0, 1, 0],
                                    [1, 1, 1]])
            case 'Z':
                pattern = np.array([[0, 1, 1],
                                    [1, 1, 0]])
            case _:
                raise ValueError(f'{name} is not a recognised pattern specifier.')
        return Piece(pattern, column)


def main(line: str) -> int:
    """
    Simulate a game of simplified Tetris with the sequence of pieces specified
    in `line` and return the height of the highest occupied cell when the game
    is finished.
    """
    grid = Grid()
    pieces = parse_line(line)
    for piece in pieces:
        grid.add_piece(piece)
    return grid.highest_non_empty_row()


def parse_line(line: str) -> list[Piece]:
    specifiers = line.split(',')
    return [Piece.from_specifier(specifier) for specifier in specifiers]


if __name__ == '__main__':
    for line in fileinput.input():
        print(main(line))
