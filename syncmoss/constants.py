# Constants
numro = 50   # max number of model rows in the parameters/results tables
numco = 27   # max number of parameter columns per row (27 = the SCDW slot count)

# ===========================================================================
# 57Fe physics constants -- the single source for models.py, models_positions.py,
# Calibration.py and instrumental_io.py.
#
# The g-factors are in good agreement with the experimentally measured alpha-Fe
# line positions: the ratios they predict reproduce +-5.3123 / +-3.0760 /
# +-0.8397 mm/s to within 0.0006 mm/s. The hyperfine field is taken from
# experimental work as well, so the nuclear magneton can be defined from those
# two -- and it deviates from the experimentally found magneton by only 0.29%.
# `mun` is therefore DERIVED, and locked to the field through
# ALPHA_FE_FIELD / TESLA_PER_MMS == 2 * ALPHA_FE_V_OUTER: do not substitute
# MUN_CODATA for it without also changing ALPHA_FE_FIELD.
#
# Sources
#   ALPHA_FE_FIELD, ALPHA_FE_V_OUTER -- C. E. Violet, D. N. Pipkorn, "Mossbauer
#       line positions and hyperfine interactions in alpha iron", J. Appl. Phys.
#       42, 4339 (1971): H_hf = -(330.4 +/- 0.3) kOe at 298 K and the line
#       positions of the NBS SRM 1541 alpha-iron foil.
#   MUN_CODATA -- CODATA-2014 nuclear magneton (reference only, no formula uses it).
#   E0 -- 57Fe Mossbauer transition energy.
#   NAT_WIDTH -- natural line width of the 14.4 keV level: 4.7e-9 eV
#       (R. Roehlsberger) == 0.0978 mm/s. The rounded 0.098 mm/s is the value
#       used everywhere (start values, bounds, grid floors and the thickness
#       prefactor), so it is the primary form and G_nat follows from it.
# ===========================================================================
NAT_WIDTH = 0.098               # natural line width (FWHM), mm/s
E0 = 14412                      # resonance energy, eV
E0_J = E0 * 1.602176634 * 10**-19
c = 2.99792458 * 10 ** 11       # speed of light, mm/s
G_nat = NAT_WIDTH / c * E0      # the same natural width in eV: 4.711e-9
d = 0.005                       # area density
Fa = 0.54676                    # fraction of resonance absorption
etto = 1                        # percent of 57Fe
sigma = 2.464 * 10 ** -22       # max resonance cross section

ggr = 0.18121                   # nuclear g-factor, I = 1/2 ground state
gex = -0.10353                  # nuclear g-factor, I = 3/2 14.4 keV state
MUN_CODATA = 5.050783699*10**-27  # nuclear magneton, J/T
ALPHA_FE_V_OUTER = 5.3123       # alpha-Fe outer line at 298 K, mm/s
ALPHA_FE_FIELD = 33.04          # alpha-Fe hyperfine field at 298 K, T

# A Zeeman line (m_gr -> m_ex) sits at -(gex*m_ex - ggr*m_gr) * MMS_PER_T_PER_G * H,
# i.e. at -+(ggr-3*gex)/2 for lines 1,6, -+(ggr-gex)/2 for 2,5, -+(ggr+gex)/2 for 3,4.
mun = 2 * ALPHA_FE_V_OUTER / (ALPHA_FE_FIELD * (ggr - 3 * gex)) * E0_J / c
                                                        # 5.036142e-27 J/T
MMS_PER_T_PER_G = mun * c / E0_J                        # 0.6538514 mm/s per T per g
LINE_SHIFT_16 = (ggr - 3 * gex) / 2 * MMS_PER_T_PER_G   # 0.1607831 mm/s per T
LINE_SHIFT_25 = (ggr - gex) / 2 * MMS_PER_T_PER_G       # 0.0930894 mm/s per T
LINE_SHIFT_34 = (ggr + gex) / 2 * MMS_PER_T_PER_G       # 0.0253958 mm/s per T
TESLA_PER_MMS = 1 / (2 * LINE_SHIFT_16)                 # 3.1097641 T per mm/s of the
                                                        # full (line 1<->6) splitting
LINE_RATIO_25 = LINE_SHIFT_25 / LINE_SHIFT_16           # 0.5789752
LINE_RATIO_34 = LINE_SHIFT_34 / LINE_SHIFT_16           # 0.1579504

# Linear polarization degree of the SMS (synchrotron) beam, 0..1. Only a default:
# it is GUI-editable via Supp -> "Set polarization" and travels to the models as
# the `pol` / `sms_pol` argument. 0.98 is a realistic synchrotron beam.
SMS_POL_DEFAULT = 0.98

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