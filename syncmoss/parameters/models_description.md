# SYNCmoss models description

Every row of the model dropdown and its parameters, named as in the table and in
table order. Units: mm/s, T, degrees. **Ctrl+F** finds a word. Longer notes on
the models: *Supp -> Help (extended)*, section 13; links, `par` numbering and
`Model_<range>`: *Supp -> Help*.

- Angles: $\theta_k$ from the beam direction $\mathbf{k}$, $\varphi_h$ the
  azimuth from the polarization direction $\mathbf{h}$.
- Angles, texture ($A$, $Am$, $Ah$) and grid parameters start **fixed**.

---

## 1. Baseline rows

### baseline
Background of the first spectrum, polynomials in the velocity $v$ around their
own centres; $N_0$ multiplies the transmission, $N_1$ is added to it:

$$
N_0 = N_s\left[1 + \frac{lin_s}{10^2}(v - O_s) + \frac{c^2_s}{10^4}(v - O_s)^2\right]
$$

$$
N_1 = N_{nr}\left[1 + \frac{lin_{nr}}{10^2}(v - O_{nr}) + \frac{c^2_{nr}}{10^4}(v - O_{nr})^2\right]
$$

Parameters:
- Ns: source-side baseline level.
- Os: source-side centre of the linear and quadratic terms.
- c²s: source-side quadratic term.
- lins: source-side linear term.
- Nnr: non-resonant baseline level.
- Onr: non-resonant centre of the linear and quadratic terms.
- c²nr: non-resonant quadratic term.
- linnr: non-resonant linear term.

Onr, c²nr and linnr start linked to Os, c²s and lins (see *Help*, section 6).

### Nbaseline
Baseline of one more spectrum in a SIMULTANEOUS fit: $n$ spectra need $n-1$
Nbaseline rows.

Parameters are the same as baseline:
- Ns, Os, c²s, lins, Nnr, Onr, c²nr, linnr.

---

## 2. Hyperfine component models

### Singlet
Single line (isotropic).

Parameters:
- T: effective thickness.
- $\delta$: central shift.
- L: Lorentzian width (typically fixed near natural width).
- G: Gaussian width.

### Doublet
Quadrupole doublet.

Parameters:
- T: effective thickness.
- $\delta$: central shift.
- $\varepsilon$: quadrupole shift — HALF the splitting; the lines sit at
  $\delta \pm \varepsilon$.
- L: Lorentzian width.
- G: Gaussian width (of line 1).
- $\theta_k$: polar angle of the EFG axis (from the beam direction).
- $\varphi_h$: azimuth of the EFG axis (from the polarization direction).
- A: uniaxial texture order parameter, $A\in[-0.5, 1]$.
- G2/G1: Gaussian width of line 2 as a multiple of G (1 = equal).

### Sextet
Magnetic sextet.

Parameters:
- T: effective thickness.
- $\delta$: central shift.
- $\varepsilon$: quadrupole shift — lines 1, 6 move by $+\varepsilon$, the
  other four by $-\varepsilon$.
- H: hyperfine magnetic field.
- L: Lorentzian width.
- G: Gaussian width.
- $\theta_k$: polar angle of the hyperfine-field axis.
- $\varphi_h$: azimuth of the hyperfine-field axis.
- A: uniaxial texture order parameter.
- Am: magnetic polar-order parameter (Faraday term).
- a+: extra splitting of lines 1 and 6 (1 by $-a_+$, 6 by $+a_+$).
- a-: the same for the inner lines (2, 4 by $+a_-$, 3, 5 by $-a_-$).
- GH: Gaussian spread of H, in T (outer lines broaden most).
- I1/I3: intensity ratio of lines 1, 6 to lines 3, 4; 3 = isotropic.

### MDGD
Magnetic sextet with correlated multidimensional distribution broadening.

Parameters:
- T, $\delta$, $\varepsilon$, H, L, G: as in Sextet.
- GH: spread of H, as in Sextet.
- D$\delta\varepsilon$: correlation between $\delta$ and $\varepsilon$.
- D$\delta$H: correlation between $\delta$ and H.
- D$\varepsilon$H: correlation between $\varepsilon$ and H.
- $\theta_k$: axis polar angle.
- $\varphi_h$: axis azimuth.
- A: uniaxial texture order parameter.
- Am: magnetic polar-order parameter.
- a+, a-: line-position corrections, as in Sextet.
- I1/I3: outer-to-inner intensity ratio, as in Sextet.

### Relax_MS
Many-state superparamagnetic relaxation of a particle of spin S.

Parameters:
- T: effective thickness.
- $\delta$: central shift.
- $\varepsilon$: quadrupole contribution.
- H: saturation hyperfine field (state $m$ sees $H\,m/S$).
- L: Lorentzian width; no Gaussian — the relaxation is the broadening.
- $\theta_k$: axis polar angle.
- $\varphi_h$: axis azimuth.
- A: uniaxial texture order parameter.
- R: relaxation rate (small = static sextet, large = collapsed line).
- alfa: anisotropy barrier in units of $k_BT$ (0 = free hopping).
- S: spin of the particle ($2S+1$ states); structural, never fitted.

### Relax_2S
Two-state (Blume) relaxation between two hyperfine sets.

Parameters:
- T: effective thickness.
- $\delta_1$, $\varepsilon_1$, H1: state 1 hyperfine set.
- $\delta_2$, $\varepsilon_2$, H2: state 2 hyperfine set (H2 = -H1: a reversing
  moment).
- L: Lorentzian width.
- $\theta_k$: axis polar angle.
- $\varphi_h$: axis azimuth.
- A: uniaxial texture order parameter.
- Am: magnetic polar-order parameter.
- $\Omega_{12}$: transition rate, mm/s ($\ll L$ slow, $\gg$ splitting fast).
- P1/P2: population ratio of the two states.

### Hamiltonian
Full Hamiltonian (magnetic + quadrupole, any orientation) of a mosaic crystal:
$A = Am = Ah = 1$ a single crystal, $A = Am = Ah = 0$ a powder.

Parameters:
- T: effective thickness.
- $\delta$: central shift.
- Q: quadrupole coupling scale.
- H: magnetic field.
- L: Lorentzian width.
- G: Gaussian width.
- $\eta$: EFG asymmetry parameter.
- $\theta_H$, $\varphi_H$: hyperfine field direction in PAS.
- $\theta$, $\varphi$: PAS direction of the lab reference axis (**h** for SMS,
  the beam **k** for CMS).
- $\alpha_k$: beam rotation about **h** (SMS only; keep it fixed at 0 in CMS).
- A: mosaic tilt order (1 = single crystal, 0 = isotropic tilt).
- Am: polar order of that tilt (Faraday term).
- Ah: azimuthal order about the reference axis (0 = fiber texture).

### ASM
Anharmonic spin modulation (cycloid-like) model.

Parameters:
- T: effective thickness.
- $\delta$: central shift.
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
- Num: sampling points per modulation period (accuracy↔speed); structural.
- I13: intensity ratio of lines 1, 6 to lines 3, 4, as in Sextet.
- $\omega$: angle of the cycloid plane about the easy axis.

### SCDW
Spin/charge density wave: field, central shift and quadrupole shift modulated
along the wave; the spin axis is fixed at $(\theta_k, \varphi_h)$.

Parameters (in table order):
- T: effective thickness.
- $\delta$: base central shift $\delta_0$.
- $\varepsilon$: base quadrupole shift $\varepsilon_0$.
- H0: base hyperfine field.
- L: Lorentzian width.
- G: Gaussian width.
- $\theta_k$: spin-axis polar angle (beam frame).
- $\varphi_h$: spin-axis azimuth.
- A: uniaxial texture order parameter.
- Am: magnetic polar-order parameter, as in Sextet.
- $K_\delta H$: central-shift–field correlation (mm/s per field unit).
- $K_\varepsilon H$: quadrupole–field correlation (mm/s per field unit).
- $\Phi$: CDW phase (deg).
- h1, h3, …, h15: eight odd SDW field harmonics (field units); unused ones stay 0.
- d2, d4, d6, d8: four even CDW central-shift harmonics (mm/s); unused ones stay 0.
- N/Γ: grid steps per line width (accuracy↔speed, default 4); structural.
- I13: intensity-ratio control between outer and inner groups.

---

## 3. Presets and utility component rows

### Be
Doublet preset of the beamline optics impurity, from `Be.txt` (*Supp → Set
parameters of Be*). Parameters as Doublet:
- T, $\delta$, $\varepsilon$, L, G, $\theta_k$, $\varphi_h$, A, G2/G1.

### KB_nano
Doublet preset of the Nanoscope impurity, from `KB.txt` (*Supp → Set
parameters of KB*). Parameters as Doublet:
- T, $\delta$, $\varepsilon$, L, G, $\theta_k$, $\varphi_h$, A, G2/G1.

### Layer
Boundary between two layers of a stack: the components above and below it are
propagated as separate layers. No parameters.

### Library, Load model, Load expression, Copy, Paste, Insert, Delete
Not models: they act on their row. `Library` and `Load model` ADD a model there
(baseline kept, references re-indexed), `Load expression` an Expression;
`Copy`/`Paste` copy a row; `Insert`/`Delete` add/remove a row (references
renumbered).

---

## 4. Expression/distribution helper rows

`Distr`, `Corr` and `Recon` act on the model row ABOVE them; `par` is the
column number in that row (1 = its second parameter, $\delta$ mostly). The
parameter `par` selects gets a grey frame and becomes read-only.

### Distr
Distribution of one parameter of the model above; several rows give a
multidimensional one.

Parameters:
- par: which parameter of the model above is distributed.
- L: left bound of distribution axis.
- R: right bound of distribution axis.
- Num: number of grid points.
- Probability density function: expression in X (the axis), e.g. `exp(-X**2/2)`.

### Corr
Makes a second parameter follow the distributed one; only under a
`Distr`/`Corr`/`Recon`.

Parameters:
- par: which parameter of the model above is driven.
- Dependency function: expression in X (the distributed value), e.g. `0.3*X`.

### Recon
Distribution with a FITTED shape.

Parameters:
- par, L, R, Num: as for `Distr`.
- D_dif: weight of the first-derivative smoothness penalty, 0…1 (0 = free).
- D_dif2: the same for the second derivative.
- weights: the reconstructed distribution, comma-separated; empty = uniform
  start, the fit writes the result here.

### Expression
A formula in `p[i]`; its result is a parameter of its own — link to it with
`=[X,Y]`.

Parameters:
- Expression: formula text, for example `p[0]` or `p[5]*0.5`.

Ready-made expressions are inserted with *Load expression*.

Expressions are calculated in list order - thus, expression should not refer to another expression which stays below (but it can refer to any parameter)

Avoid chain links in expression - if some parameter number `Z` is linked like `=[X,Y]` then expressions should not use `p[Z]` but `p[X]`.

### Variables
Free scalar values to use in expressions and links; the row is as wide as the
table (currently V1 … V27).

Parameters:
- V1 … V27: user-defined scalar values.

---

## 5. Texture and polarization

- A: second-moment alignment ($A=0$ random powder, the default; $A=1$ aligned
  along the axis; $A=-1/2$ axis in the perpendicular plane).
- Am: magnetic polar order, in Sextet, MDGD, Relax_2S, SCDW, Hamiltonian; a
  thick-only effect.
- Ah: azimuthal order (Hamiltonian only).
- SMS: linearly polarized readout (*Supp -> Set SMS polarization*, default 0.98);
  CMS: unpolarized.
