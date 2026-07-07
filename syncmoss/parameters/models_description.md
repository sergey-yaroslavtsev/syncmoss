# SYNCmoss models description

This file summarizes all model rows available in SYNCmoss, their parameters,
and practical meaning.

Notes:
- Parameter names here follow the GUI labels exactly.
- Velocity-like units are mm/s, magnetic field is T, angles are degrees.
- For polarized thick models, orientation angles are defined as:
  - $\theta_k$: angle from beam direction $\mathbf{k}$.
  - $\varphi_h$: azimuth from polarization direction $\mathbf{h}$.

---

## 1. Baseline rows

### baseline
Background polynomial for the first spectrum (global baseline row).

Parameters:
- Ns: source-side baseline level.
- Os: source-side offset.
- c2s: source-side quadratic term.
- lins: source-side linear term.
- Nnr: non-resonant baseline level.
- Onr: non-resonant offset.
- c2nr: non-resonant quadratic term.
- linnr: non-resonant linear term.

### Nbaseline
Baseline row used for sequence/simultaneous workflows (one per extra spectrum).

Parameters are the same as baseline:
- Ns, Os, c2s, lins, Nnr, Onr, c2nr, linnr.

---

## 2. Hyperfine component models

### Singlet
Single Lorentzian/Voigt-like resonance line (isotropic).

Parameters:
- T: effective thickness / intensity scale.
- $\delta$: isomer shift.
- L: Lorentzian width (typically fixed near natural width).
- G: Gaussian broadening.

### Doublet
Quadrupole doublet in polarized thick-matrix formalism.

Parameters:
- T: effective thickness.
- $\delta$: isomer shift.
- $\varepsilon$: quadrupole splitting.
- L: Lorentzian width.
- G: Gaussian width.
- $\theta_k$: axis polar angle (from beam direction).
- $\varphi_h$: axis azimuth (from polarization direction).
- A: uniaxial (fiber) texture order parameter, $A\in[-0.5, 1]$.
- G2/G1: ratio of Gaussian widths for line 2 and line 1.

### Sextet
Magnetic sextet in polarized thick-matrix formalism.

Parameters:
- T: effective thickness.
- $\delta$: isomer shift.
- $\varepsilon$: quadrupole contribution.
- H: hyperfine magnetic field.
- L: Lorentzian width.
- G: Gaussian width.
- $\theta_k$: axis polar angle.
- $\varphi_h$: axis azimuth.
- A: uniaxial texture order parameter.
- A_m: magnetic polar-order parameter (Faraday-active contribution).
- a+: positive branch shift correction.
- a-: negative branch shift correction.
- GH: field-distribution width parameter.
- I1/I3: outer-to-inner line intensity ratio control.

### MDGD
Magnetic sextet with correlated multidimensional distribution broadening.

Parameters:
- T, $\delta$, $\varepsilon$, H, L, G: same meaning as Sextet.
- GH: field-distribution width term.
- Dde: correlation/spread term between $\delta$ and $\varepsilon$.
- DdH: correlation/spread term between $\delta$ and H.
- DeH: correlation/spread term between $\varepsilon$ and H.
- $\theta_k$: axis polar angle.
- $\varphi_h$: axis azimuth.
- A: uniaxial texture order parameter.
- A_m: magnetic polar-order parameter.
- a+, a-: shift corrections.
- I1/I3: outer-to-inner intensity ratio control.

### Relax_MS
Many-state magnetic relaxation model.

Parameters:
- T: effective thickness.
- $\delta$: isomer shift.
- $\varepsilon$: quadrupole contribution.
- H: magnetic field scale.
- L: Lorentzian width.
- $\theta_k$: axis polar angle.
- $\varphi_h$: axis azimuth.
- A: uniaxial texture order parameter.
- R: relaxation-rate parameter.
- alfa: distribution/shape factor for relaxation kernel.
- S: numerical resolution parameter (typically fixed).

### Relax_2S
Two-state Blume relaxation model.

Parameters:
- T: effective thickness.
- $\delta_1$, $\varepsilon_1$, H1: state 1 hyperfine set.
- $\delta_2$, $\varepsilon_2$, H2: state 2 hyperfine set.
- L: Lorentzian width.
- $\theta_k$: axis polar angle.
- $\varphi_h$: axis azimuth.
- A: uniaxial texture order parameter.
- A_m: magnetic polar-order parameter.
- $\Omega_{12}$: transition rate between states.
- P1/P2: state-population ratio.

### Hamilton_mc
Single-crystal Hamiltonian model (magnetic + quadrupole, full orientation).

Parameters:
- T: effective thickness.
- $\delta$: isomer shift.
- Q: quadrupole coupling scale.
- H: magnetic field.
- L: Lorentzian width.
- G: Gaussian width.
- $\eta$: EFG asymmetry parameter.
- $\theta_H$, $\varphi_H$: hyperfine field direction in PAS.
- $\theta$, $\varphi$: geometry angles (source-mode dependent).
- $\alpha_k$: beam-rotation angle used in thick SMS geometry.

### Hamilton_pc
Powder/averaged Hamiltonian variant.

Parameters:
- T: effective thickness.
- $\delta$: isomer shift.
- Q: quadrupole coupling scale.
- H: magnetic field.
- L: Lorentzian width.
- G: Gaussian width.
- $\eta$: EFG asymmetry parameter.
- $\theta_H$, $\varphi_H$: hyperfine field orientation.

### ASM
Anharmonic spin modulation (cycloid-like) model.

Parameters:
- T: effective thickness.
- $\delta$: isomer shift.
- $\varepsilon_m$: modulated quadrupole term.
- $\varepsilon_l$: lattice/static quadrupole term.
- His: isotropic magnetic field component.
- Han: anharmonic modulation field amplitude.
- L: Lorentzian width.
- G: Gaussian width.
- m: elliptic/anharmonicity parameter.
- $\theta_k$: modulation-axis polar angle.
- $\varphi_h$: modulation-axis azimuth.
- A: uniaxial texture order parameter.
- Num: number of sampling points over one modulation period.
- I13: intensity-ratio control between outer and inner groups.

### Average_H (legacy/advanced)
Field-averaging model retained for compatibility.

Parameters:
- T: effective thickness.
- $\delta$: isomer shift.
- $\varepsilon$: quadrupole splitting.
- Hin: internal field.
- L: Lorentzian width.
- G: Gaussian width.
- Hex: external field.
- K: anisotropy-like coefficient.
- J: coupling/sign coefficient.
- $\theta$: orientation angle.
- N: numerical averaging resolution.

---

## 3. Presets and utility component rows

### Be
Preset impurity component (pre-filled Doublet-like row loaded from Be.txt).

Parameters are identical to Doublet:
- T, $\delta$, $\varepsilon$, L, G, $\theta_k$, $\varphi_h$, A, G2/G1.

### KB_nano
Preset impurity component (pre-filled Doublet-like row loaded from KB.txt).

Parameters are identical to Doublet:
- T, $\delta$, $\varepsilon$, L, G, $\theta_k$, $\varphi_h$, A, G2/G1.

### Layer
Layer boundary marker for non-commuting thick-matrix propagation.

Parameters:
- none.

Behavior:
- Components above and below Layer are exponentiated as separate layer matrices.
- For scalar-only models this behaves like a no-op.

---

## 4. Expression/distribution helper rows

### Distr
Adds a parameter distribution to the preceding physical model.

Parameters:
- par: index of model parameter to distribute.
- L: left bound of distribution axis.
- R: right bound of distribution axis.
- Num: number of grid points.
- Probability density function: expression in variable X.

### Corr
Adds parameter dependence/correlation to the preceding model.

Parameters:
- par: index of controlled parameter.
- Dependency function: expression in variable X.

### Expression
Adds a free algebraic expression evaluated on parameter array p.

Parameters:
- Expression: formula text, for example p[0] or p[5]*0.5.

### Variables
Helper row with named placeholders V1..V17.

Parameters:
- V1 ... V17: user-defined scalar values for referencing.

---

## 5. Texture and polarization quick guide

For thick polarized models, matrix transmission is used:

$$
T(v) = \left[\exp\left(-\hat{\Sigma}(v)\right)\right]_{11}
$$

with source-mode-dependent readout for partial/unpolarized beams:

$$
C_a(v) = \operatorname{tr}\!\left[\exp(-\hat{\Sigma}(v))\,\rho\right]
$$

Texture parameters:
- A controls second-moment alignment ($A=0$ random powder, $A=1$ aligned).
- A_m controls first-moment magnetic polar order in Faraday-active models.

Practical fitting advice:
- Start with A = 0 (powder-like) unless strong texture is expected.
- Keep orientation angles fixed initially, then release if residuals suggest anisotropy.
- For Sextet/MDGD/Relax_2S, release A_m only when data quality supports it.

---

## 6. Source mode notes

- SMS mode: linearly polarized readout (default near fully polarized).
- CMS mode: unpolarized half-trace readout.
- Thick models are valid in both modes, but angular sensitivity differs.

For detailed thick-model derivations and explicit matrices, see thick_models.md.
