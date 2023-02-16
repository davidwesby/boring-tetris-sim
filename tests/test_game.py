import numpy as np
import pytest

from tetris import main, parse_line, Piece


@pytest.mark.parametrize('test_input, expected_output', [
    ('Q0', 2),
    ('Q0,Q1', 4),
    ('Q0,Q2,Q4,Q6,Q8', 0),
    ('Q0,Q2,Q4,Q6,Q8,Q1', 2),
    ('Q0,Q2,Q4,Q6,Q8,Q1,Q1', 4),
    ('I0,I4,Q8', 1),
    ('I0,I4,Q8,I0,I4', 0),
    ('L0,J2,L4,J6,Q8', 2),
    ('L0,Z1,Z3,Z5,Z7', 2),
    ('T0,T3', 2),
    ('T0,T3,I6,I6', 1),
    ('I0,I6,S4', 1),
    ('T1,Z3,I4', 4),
    ('L0,J3,L5,J8,T1', 3),
    ('L0,J3,L5,J8,T1,T6', 1),
    ('L0,J3,L5,J8,T1,T6,J2,L6,T0,T7', 2),
    ('L0,J3,L5,J8,T1,T6,J2,L6,T0,T7,Q4', 1),
    ('S0,S2,S4,S6', 8),
    ('S0,S2,S4,S5,Q8,Q8,Q8,Q8,T1,Q1,I0,Q4', 8),
    ('L0,J3,L5,J8,T1,T6,S2,Z5,T0,T7', 0),
    ('Q0,I2,I6,I0,I6,I6,Q2,Q4', 3),
])
def test_main(test_input: str, expected_output: int):
    assert main(test_input) == expected_output


@pytest.mark.parametrize('test_input, expected_output', [
    ('L0,J2,L4', [Piece(np.array([[1, 1], [1, 0], [1, 0]]), 0),
                  Piece(np.array([[1, 1], [0, 1], [0, 1]]), 2),
                  Piece(np.array([[1, 1], [1, 0], [1, 0]]), 4)]),
    ('L1',       [Piece(np.array([[1, 1], [1, 0], [1, 0]]), 1)]),
])
def test_parse_line(test_input, expected_output):
    assert parse_line(test_input) == expected_output
