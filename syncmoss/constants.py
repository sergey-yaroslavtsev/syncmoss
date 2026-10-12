# Constants
import math

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
# Universal constants: the SI defining constants (exact since 2019) and the
# Bohr magneton (CODATA 2018).
E_CHARGE = 1.602176634e-19      # elementary charge, C (J per eV)
H_PLANCK = 6.62607015e-34       # Planck constant, J s
K_B = 1.380649e-23              # Boltzmann constant, J/K
MU_B = 9.2740100783e-24         # Bohr magneton, J/T

NAT_WIDTH = 0.098               # natural line width (FWHM), mm/s
E0 = 14412                      # resonance energy, eV
E0_J = E0 * E_CHARGE            # resonance energy, J
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

# --- scattering-side 57Fe quantities (sms_theory.py) -----------------------
# Everything below is DERIVED from the primitives above plus two new inputs,
# ALPHA_IC and the universal hc; no 57Fe literal is repeated.
#   ALPHA_IC -- internal conversion coefficient of the 14.4 keV level.
#   HC_EV_A  -- h*c, from H_PLANCK, c and E_CHARGE (exact in SI since 2019).
ALPHA_IC = 8.56
HC_EV_A = H_PLANCK * c / E_CHARGE * 1e7                 # 12398.42 eV * Angstrom

MMPS_TO_NEV = E0 / c * 10**9            # 48.0733 neV per (mm/s): E = E0 v/c
GAMMA0_NEV = NAT_WIDTH * MMPS_TO_NEV    # 4.7112 neV, the natural width in neV
MU_N_NEV_PER_T = mun / (E0_J / E0) * 10**9              # 31.4331 neV/T
LAMBDA_A = HC_EV_A / E0                                 # 0.8603 A
K_VEC_A = 2 * math.pi / LAMBDA_A                        # 7.3035 1/A
R_E_A = 2.8179403262 * 10**-5                           # classical electron radius, A
# Maximum resonant cross-section from the level scheme:
#   sigma_0 = 2 pi lambdabar^2 (2 I_e + 1)/(2 I_g + 1) / (1 + alpha_ic)
# equals `sigma` above (2.464e-22 m^2) to 0.02 %; sms_theory.selftest() checks it.
SIGMA0_A2 = (2 * math.pi * (LAMBDA_A / (2 * math.pi))**2
             * (2 * 1.5 + 1) / (2 * 0.5 + 1) / (1 + ALPHA_IC))   # 0.024644 A^2

# --- lattice-dynamics and lifetime quantities (the Expressions/ formulas) ----
# The predefined expressions hold these as plain numbers (their text is
# evaluated, it cannot import), written from the values here.
#   M_FE57_U  -- atomic mass of 57Fe, u (AME2020: 56.935 392 84(29) u).
#   U_C2_EV   -- energy equivalent of the atomic mass unit, eV
#                (CODATA 2018: 931.494 102 42(28) MeV).
K_B_EV = K_B / E_CHARGE                                 # 8.617333e-5 eV/K
HBAR_EV_S = H_PLANCK / (2 * math.pi) / E_CHARGE         # 6.582120e-16 eV s
M_FE57_U = 56.93539284
U_C2_EV = 931.49410242e6
MC2_EV = M_FE57_U * U_C2_EV                             # 5.303498e10 eV, rest energy of the atom
E_RECOIL_EV = E0**2 / (2 * MC2_EV)                      # 1.958e-3 eV, free-atom recoil energy

# Linear polarization degree of the SMS (synchrotron) beam, 0..1. Only a default:
# it is GUI-editable via Supp -> "Set polarization" and travels to the models as
# the `pol` / `sms_pol` argument. 0.98 is a realistic synchrotron beam.
SMS_POL_DEFAULT = 0.98

# Where users reach the author: the Supp -> "Contact the author" entry, the
# bug-report window raised by error_reporter and the README all use these.
AUTHOR_EMAIL = 'sergey.yaroslavtsev@esrf.fr'
ISSUES_URL = 'https://github.com/sergey-yaroslavtsev/syncmoss/issues'

# Number of baseline parameters
number_of_baseline_parameters = 8

# Default color sequence for plotting models
model_colors = ['red', 'blue', 'cyan', 'yellow', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato', 'white', 'silver', 'lightgreen', 'pink']

# An Nbaseline row is shown light grey: it opens the next spectrum's section,
# it is not a component. 'silver' is a colour every .mdl reader already knows.
NBASELINE_COLOR = 'silver'

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