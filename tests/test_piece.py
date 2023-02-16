import numpy as np
import pytest

from tetris import Piece


@pytest.mark.parametrize('test_input, expected_output', [
    ('Q4', Piece(np.array([[1, 1], [1, 1]]), 4)),
    ('L1', Piece(np.array([[1, 1], [1, 0], [1, 0]]), 1)),
])
def test_piece_from_specifier(test_input: str, expected_output: Piece):
    assert Piece.from_specifier(test_input) == expected_output
