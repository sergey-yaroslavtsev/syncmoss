"""Exclusion regions: the text the user types, the mask of fitted points, where a
clicked boundary goes, and the file Save/Load write and read.

Pure Python/NumPy -- no Qt.
"""
import numpy as np
import pytest

from syncmoss import exclusion_regions as er


# --- the text ---------------------------------------------------------------

def test_regions_are_read_sorted_whatever_order_the_ends_are_typed_in():
    assert er.parse("1:2; -3:-2") == ((-3.0, -2.0), (1.0, 2.0))
    assert er.parse("-2:-3") == ((-3.0, -2.0),)


def test_overlapping_and_touching_regions_become_one():
    assert er.parse("1:3; 2:4") == ((1.0, 4.0),)
    assert er.parse("1:2; 2:3") == ((1.0, 3.0),)
    assert er.parse("1:5; 2:3") == ((1.0, 5.0),)


def test_blank_text_and_empty_pieces_mean_no_regions():
    assert er.parse("") == ()
    assert er.parse("   ") == ()
    assert er.parse("1:2;") == ((1.0, 2.0),)
    assert er.parse("1:2;;3:4") == ((1.0, 2.0), (3.0, 4.0))


@pytest.mark.parametrize("text", [
    "1:2, 3:4",        # ',' between regions
    "-3,5:-2",         # a decimal comma
    "1-2",             # no ':'
    "1:2:3",           # two ':'
    "a:2",             # not a number
    "1:",              # one end missing
    "2:2",             # an empty region
    "inf:2",
    "1:nan",
])
def test_text_that_is_not_regions_is_refused(text):
    with pytest.raises(ValueError):
        er.parse(text)


def test_the_written_text_reads_back_the_same_regions():
    regions = er.parse("-3.05:-1.95; 0.95:2.05; 10:12.5")
    text = er.format_regions(regions)
    assert text == "-3.05:-1.95; 0.95:2.05; 10:12.5"
    assert er.parse(text) == regions


def test_numbers_are_written_in_full_not_rounded():
    regions = ((-2.0352187, 1.0 / 3.0),)
    assert er.parse(er.format_regions(regions)) == regions


# --- which points are fitted -----------------------------------------------

def test_both_ends_of_a_region_are_excluded():
    v = np.array([-3.0, -2.5, -2.0, -1.5, 0.0, 1.0, 1.5, 2.0, 2.5])
    keep = er.kept(v, ((-2.5, -2.0), (1.0, 2.0)))
    assert keep.tolist() == [True, False, False, True, True, False, False, False, True]


def test_the_direction_of_the_velocity_axis_does_not_matter():
    v = np.linspace(-10, 10, 101)
    regions = ((-3.0, -2.0), (1.0, 2.0))
    assert np.array_equal(er.kept(v[::-1], regions), er.kept(v, regions)[::-1])


def test_no_regions_keep_every_point():
    assert er.kept(np.linspace(-1, 1, 7), ()).all()


# --- where a clicked boundary goes -----------------------------------------

def test_the_fewest_decimals_strictly_inside_closest_to_the_target():
    assert er.shortest_decimal_in(3.0, 7.0, 4.4) == 4.0
    assert er.shortest_decimal_in(-2.0352, -1.9961, -2.01734) == -2.0
    assert er.shortest_decimal_in(0.11, 0.19, 0.17) == 0.17
    # a target outside the interval takes the candidate nearest to it
    assert er.shortest_decimal_in(10.0, 10.039, 10.6) == 10.03


def test_the_ends_themselves_are_never_the_answer():
    # 1 and 2 would be the shortest, but they are the points themselves
    assert er.shortest_decimal_in(1.0, 2.0, 1.0) == 1.1


@pytest.mark.parametrize("grid, click, boundary", [
    (np.arange(-10, 10.5, 0.5), -2.2, -2.2),          # step 0.5: one decimal
    (np.linspace(-10, 10, 513), -2.01734, -2.0),      # step 0.039: -2 lies between the points
    (np.linspace(0, 1, 101), 0.1234, 0.123),          # step 0.01: three decimals
])
def test_a_boundary_has_the_decimals_the_point_spacing_needs(grid, click, boundary):
    value = er.boundary_between_points(grid, click)
    assert value == boundary
    i = np.searchsorted(grid, click)
    assert grid[i - 1] < value < grid[i]          # between the two points around the click


def test_a_boundary_clicked_beyond_the_spectrum_stays_within_one_step_of_it():
    v = np.linspace(-10, 10, 513)
    step = v[1] - v[0]
    right = er.boundary_between_points(v, 13.0)
    left = er.boundary_between_points(v, -13.0)
    assert 10.0 < right < 10.0 + step
    assert -10.0 - step < left < -10.0


def test_a_reversed_grid_gives_the_same_boundary():
    v = np.linspace(-10, 10, 513)
    assert er.boundary_between_points(v[::-1], 1.234) == er.boundary_between_points(v, 1.234)


def test_without_points_a_boundary_is_within_half_a_pixel():
    assert er.boundary_at_pixel(1.23456, 0.02) == 1.23
    assert er.boundary_at_pixel(1.23456, 0.2) == 1.2      # a wider pixel, fewer decimals


def test_a_grid_of_one_point_gives_no_boundary():
    assert er.boundary_between_points([1.0], 0.5) is None


# --- the file ---------------------------------------------------------------

def test_the_file_reads_back_the_regions_it_was_written_with(tmp_path):
    path = tmp_path / "Fe_exclusion.txt"
    regions = er.parse("-3.05:-1.95; 0.95:2.05")
    er.write_file(str(path), regions)
    assert path.read_text(encoding='utf-8').splitlines()[0].startswith('#')
    assert er.read_file(str(path)) == regions


def test_a_file_of_several_lines_and_comments_is_read_as_one_list(tmp_path):
    path = tmp_path / "hand_written.txt"
    path.write_text("# my regions\n-3:-2\n\n1:2; 5:6\n# end\n", encoding='utf-8')
    assert er.read_file(str(path)) == ((-3.0, -2.0), (1.0, 2.0), (5.0, 6.0))


def test_a_bad_file_is_refused(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1,5:2\n", encoding='utf-8')
    with pytest.raises(ValueError):
        er.read_file(str(path))
