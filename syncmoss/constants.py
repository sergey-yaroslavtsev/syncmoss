# Constants
numro = 50   # max number of model rows in the parameters/results tables
numco = 17   # max number of parameter columns per row

# Number of baseline parameters
number_of_baseline_parameters = 8

# Default color sequence for plotting models
model_colors = ['red', 'blue', 'cyan', 'yellow', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato', 'white', 'silver', 'lightgreen', 'pink']

# Every color name a .mdl color row may contain (used to recognise the color
# line when loading model files — shared by model_io and Library_io so both
# accept the same files).
mdl_color_names = ['blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange',
                   'blueviolet', 'green', 'tomato', 'pink', 'crimson', 'orange',
                   'purple', 'brown', 'gray', 'black', 'white', 'silver', 'lightgreen']

# Background colors light enough to need black (instead of white) button text.
_light_button_colors = ('red', 'yellow', 'cyan', 'lime', 'darkorange', 'white',
                        'silver', 'lightgreen', 'pink', 'lightgray')


def contrast_text_color(color):
    """Black or white — whichever is readable on a button of that color.

    Hex colors ('#rrggbb') get black text like the other light picks; this is
    the single implementation of the check previously copy-pasted (with small
    inconsistencies) across parameters_table, results_table and model_io.
    """
    if isinstance(color, str) and (color in _light_button_colors or color.startswith('#')):
        return 'black'
    return 'white'