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
- $\theta_k$: easy (anharmonicity) axis polar angle.
- $\varphi_h$: easy (anharmonicity) axis azimuth.
- A: uniaxial texture order parameter.
- Num: number of sampling points over one modulation period.
- I13: intensity-ratio control between outer and inner groups.
- $\omega$: cycloid-plane angle about the easy axis. $(\theta_k, \varphi_h)$ fix
  the easy (anharmonicity) axis $\mathbf{u}$, which always lies IN the plane the
  moment rotates in. $\omega$ picks the second in-plane axis
  $\mathbf{v} = \cos\omega\,\mathbf{e}_\theta + \sin\omega\,\mathbf{e}_\varphi$
  (rotation of $\mathbf{v}$ around $\mathbf{u}$), so the rotation plane is
  $\mathrm{span}(\mathbf{u}, \mathbf{v})$ and its normal is
  $\mathbf{n}_c = \mathbf{u}\times\mathbf{v}$. Thus $\omega$ steers the
  plane-normal direction, but the normal is constrained perpendicular to
  $\mathbf{u}$ (it sweeps the cone about $\mathbf{u}$ as $\omega$ varies) — it is
  the single remaining orientation degree of freedom once $\mathbf{u}$ is fixed.
  $\omega = \omega_0(\theta_k, \varphi_h) =
  \operatorname{atan2}(-\sin\varphi_h,\ \cos\theta_k\cos\varphi_h)$ reproduces the
  former plane-contains-$h$ geometry ($=90°$ in the degenerate $\mathbf{u}\parallel h$ case).

### S/C_DW
Spin/charge density wave. The spin AXIS is fixed at $(\theta_k, \varphi_h)$;
only the scalar hyperfine parameters (signed field magnitude, isomer shift,
quadrupole shift) are modulated along the wave, sampled at Num points over one
period and averaged as a thick (polarized) sextet. The field keeps its SIGN, so
a reversed moment automatically swaps the line positions within the (1,6) and
(3,4) pairs. Follows the SpectrRelax SDW/CDW parameter list.

Parameters (in table order):
- T: effective thickness.
- $\delta$: base isomer shift $\delta_0$.
- $\varepsilon$: base quadrupole shift $\varepsilon_0$.
- H0: base hyperfine field.
- L: Lorentzian width.
- G: Gaussian width.
- $\theta_k$: spin-axis polar angle (beam frame).
- $\varphi_h$: spin-axis azimuth.
- A: uniaxial texture order parameter.
- A_m: magnetic polar-order parameter (as in the Sextet model) that scales the
  resolved $\sigma^\pm$ Faraday term.
- $K_\delta H$: isomer-shift–field correlation (mm/s per field unit).
- $K_\varepsilon H$: quadrupole–field correlation (mm/s per field unit).
- $\Phi$: CDW phase (deg).
- h1, h3, …, h15: eight odd SDW field harmonics (field units); unused ones stay 0.
- d2, d4, d6, d8: four even CDW isomer-shift harmonics (mm/s); unused ones stay 0.
- Num: number of sampling points over one wave period.
- I13: intensity-ratio control between outer and inner groups.

The resolved $\sigma^\pm$ (Faraday) matrices are kept per site (never the merged
symmetric form). Whether the Faraday term survives the modulation is set by the
WAVE, independently of A_m: for a balanced wave (H0 = 0 and
$K_\varepsilon H = K_\delta H = 0$) the $+$/$-$ sites are populated equally and
the Faraday contributions cancel exactly for any A_m; a non-zero base field H0
(or field correlation) leaves a real, A_m-scaled, thickness-dependent Faraday
signal. With all wave parameters zero and H0 $\ne$ 0 the model reduces to a
single (Sextet-equivalent) sextet at the same A_m.

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
