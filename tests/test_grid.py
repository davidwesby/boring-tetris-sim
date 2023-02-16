import numpy as np

from tetris import Grid, Piece


def test_grid_add_piece_empty_grid():
    grid = Grid()

    # "L" piece at column 1.
    grid.add_piece(Piece(np.array([[1, 1], [1, 0], [1, 0]]), 1))

    assert np.array_equal(grid._state.nonzero(),
                          (np.array([0, 0, 1, 2]), np.array([1, 2, 1, 1])))


def test_grid_add_piece_landing_zone_clear():
    grid = Grid()
    # "Q" piece at column 4.
    np.put(grid._state, [4, 5, 14, 15], 1)

    # Add "L" piece at column 1.
    grid.add_piece(Piece(np.array([[1, 1], [1, 0], [1, 0]]), 1))

    assert np.array_equal(grid._state.nonzero(),
                          (np.array([0, 0, 0, 0, 1, 1, 1, 2]), np.array([1, 2, 4, 5, 1, 4, 5, 1])))


def test_grid_add_piece_landing_zone_not_clear():
    grid = Grid()
    # "Q" piece at column 0.
    np.put(grid._state, [0, 1, 10, 11], 1)

    # Add "L" piece at column 1.
    grid.add_piece(Piece(np.array([[1, 1], [1, 0], [1, 0]]), 1))

    assert np.array_equal(grid._state.nonzero(),
                          (np.array([0, 0, 1, 1, 2, 2, 3, 4]), np.array([0, 1, 0, 1, 1, 2, 1, 1])))


def test_grid_add_piece_bounding_box_overlaps_with_existing_pieces():
    grid = Grid()
    # "Z" piece at column 0.
    np.put(grid._state, [1, 2, 10, 11], 1)

    # Add "Z" piece at column 2.
    grid.add_piece(Piece(np.array([[0, 1, 1], [1, 1, 0]]), 2))

    assert np.array_equal(grid._state.nonzero(),
                          (np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([1, 2, 3, 4, 0, 1, 2, 3])))


def test_grid_add_piece_completes_bottom_row():
    grid = Grid()
    # Fill all but last two columns of bottom row.
    grid._state[0, 0:8] = 1

    # Add "Q" piece at column 8.
    grid.add_piece(Piece(np.array([[1, 1], [1, 1]]), 8))

    assert np.array_equal(grid._state.nonzero(),
                          (np.array([0, 0]), np.array([8, 9])))


def test_grid_add_piece_completes_ith_row():
    grid = Grid()
    # Fill all but last two columns of bottom row.
    grid._state[1, 0:8] = 1

    # Add "Q" piece at column 8.
    grid.add_piece(Piece(np.array([[1, 1], [1, 1]]), 8))

    assert np.array_equal(grid._state.nonzero(),
                          (np.array([0, 0]), np.array([8, 9])))


def test_grid_highest_non_empty_row_empty_grid():
    grid = Grid()

    assert grid.highest_non_empty_row() == 0


def test_grid_highest_non_empty_row_non_empty_grid():
    grid = Grid()
    # Arbitrary state
    np.put(grid._state, [1, 2, 10, 11, 24, 34, 44], 1)

    assert grid.highest_non_empty_row() == 5


def test_grid_get_peaks_in_columns():
    grid = Grid()
    # Arbitrary state
    np.put(grid._state, [1, 2, 10, 11, 24, 34, 44], 1)

    assert np.array_equal(grid._get_peaks_in_columns(slice(0, 4)), np.array([2, 2, 1, 0]))


def test_grid_piece_resting_height_no_overlap():
    grid = Grid()
    # "Z" piece at column 0.
    np.put(grid._state, [1, 2, 10, 11], 1)

    # "Z" piece at column 0.
    assert grid._piece_resting_height(Piece(np.array([[0, 1, 1], [1, 1, 0]]), 0)) == 2


def test_grid_piece_resting_height_overlap():
    grid = Grid()
    # "Z" piece at column 0.
    np.put(grid._state, [1, 2, 10, 11], 1)

    # "Z" piece at column 2.
    assert grid._piece_resting_height(Piece(np.array([[0, 1, 1], [1, 1, 0]]), 2)) == 0


def test_grid_remove_complete_rows_bottom_row_complete():
    grid = Grid()
    # Full bottom row, arbitrary second row.
    grid._state[0] = 1
    np.put(grid._state, [10, 11, 18, 19], 1)

    grid._remove_complete_rows()

    assert np.array_equal(grid._state.nonzero(),
                          (np.array([0, 0, 0, 0]), np.array([0, 1, 8, 9])))


def test_grid_remove_complete_rows_ith_row_complete():
    grid = Grid()
    # Full second row, arbitrary bottom row.
    grid._state[1] = 1
    np.put(grid._state, [0, 1, 2, 3], 1)

    grid._remove_complete_rows()

    assert np.array_equal(grid._state.nonzero(),
                          (np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3])))


def test_grid_remove_complete_rows_no_rows_complete():
    grid = Grid()
    # Arbitrary state with no complete rows
    np.put(grid._state, [1, 2, 10, 11, 24, 34, 44], 1)

    grid._remove_complete_rows()

    assert np.array_equal(grid._state.nonzero(),
                          (np.array([0, 0, 1, 1, 2, 3, 4]), np.array([1, 2, 0, 1, 4, 4, 4])))
