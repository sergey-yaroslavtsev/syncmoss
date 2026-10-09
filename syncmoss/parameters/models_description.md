# SYNCmoss models description

This file summarizes every model row offered in the SYNCmoss model dropdown,
its parameters and their practical meaning. Models that cannot be selected in
the GUI are not described here.

Notes:
- Parameter names here follow the GUI labels exactly, in table order.
- Velocity-like units are mm/s, magnetic field is T, angles are degrees.
- For polarized thick models, orientation angles are defined as:
  - $\theta_k$: angle from beam direction $\mathbf{k}$.
  - $\varphi_h$: azimuth from polarization direction $\mathbf{h}$.
- The angles, the texture parameters ($A$, $Am$, $Ah$) and the numerical
  (grid/resolution) parameters start **fixed**; untick "fix" only when the data
  can support them.
- For features that are not visible in the interface (the `Model_<range>`
  simulation mode, `=[X,Y]` links, the `par` numbering of `Distr`/`Corr`/`Recon`,
  keyboard shortcuts, …) see *Supp -> Help (hidden features)*.

---

## 1. Baseline rows

### baseline
Background polynomial for the first spectrum (global baseline row). Both parts
are polynomials in the velocity $v$ around their own centre:

$$
N_0 = N_s\left[1 + \frac{lin_s}{10^2}(v - O_s) + \frac{c^2_s}{10^4}(v - O_s)^2\right]
$$

$$
N_1 = N_{nr}\left[1 + \frac{lin_{nr}}{10^2}(v - O_{nr}) + \frac{c^2_{nr}}{10^4}(v - O_{nr})^2\right]
$$

$N_0$ multiplies the transmission, $N_1$ is added to it.

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
Extra baseline row for SIMULTANEOUS fitting: one per additional spectrum, so
$n$ spectra need exactly $n-1$ Nbaseline rows. Their presence is what selects
simultaneous over sequence fitting.

Parameters are the same as baseline:
- Ns, Os, c²s, lins, Nnr, Onr, c²nr, linnr.

---

## 2. Hyperfine component models

### Singlet
Single Lorentzian/Voigt-like resonance line (isotropic).

Parameters:
- T: effective thickness / intensity scale.
- $\delta$: central shift.
- L: Lorentzian width (typically fixed near natural width).
- G: Gaussian broadening.

### Doublet
Quadrupole doublet in polarized thick-matrix formalism.

Parameters:
- T: effective thickness.
- $\delta$: central shift.
- $\varepsilon$: quadrupole shift — HALF the splitting; the two lines sit at
  $\delta \pm \varepsilon$.
- L: Lorentzian width.
- G: Gaussian width (of line 1).
- $\theta_k$: axis polar angle (from beam direction) of the EFG axis.
- $\varphi_h$: axis azimuth (from polarization direction).
- A: uniaxial (fiber) texture order parameter, $A\in[-0.5, 1]$.
- G2/G1: Gaussian width of line 2 as a multiple of G — for an asymmetric
  doublet. 1 = both lines equally broad.

### Sextet
Magnetic sextet in polarized thick-matrix formalism.

Parameters:
- T: effective thickness.
- $\delta$: central shift.
- $\varepsilon$: quadrupole shift — the outer pair (1, 6) moves by $+\varepsilon$,
  the other four lines by $-\varepsilon$.
- H: hyperfine magnetic field.
- L: Lorentzian width.
- G: Gaussian width.
- $\theta_k$: polar angle of the hyperfine-field axis.
- $\varphi_h$: azimuth of the hyperfine-field axis.
- A: uniaxial texture order parameter.
- Am: magnetic polar-order parameter (Faraday-active contribution).
- a+: extra symmetric splitting of the OUTER pair only (line 1 by $-a_+$, line 6
  by $+a_+$); the second-order/relativistic correction of lines 1 and 6.
- a-: the same for the four inner lines (2 and 4 by $+a_-$, 3 and 5 by $-a_-$).
- GH: Gaussian spread of H, in T. It is added in quadrature to G with each line's
  own field sensitivity, so the outer lines broaden most — a field distribution,
  not an extra uniform width.
- I1/I3: intensity ratio of the outer (1, 6) to the inner (3, 4) lines; 3 is the
  isotropic 3:2:1 case. The middle pair (2, 5) keeps its isotropic weight and the
  total area is conserved.

### MDGD
Magnetic sextet with correlated multidimensional distribution broadening.

Parameters:
- T, $\delta$, $\varepsilon$, H, L, G: same meaning as Sextet.
- GH: field-distribution width term (as in Sextet).
- D$\delta\varepsilon$: correlation between $\delta$ and $\varepsilon$.
- D$\delta$H: correlation between $\delta$ and H.
- D$\varepsilon$H: correlation between $\varepsilon$ and H.
- $\theta_k$: axis polar angle.
- $\varphi_h$: axis azimuth.
- A: uniaxial texture order parameter.
- Am: magnetic polar-order parameter.
- a+, a-: line-position corrections (as in Sextet).
- I1/I3: outer-to-inner intensity ratio (as in Sextet).

### Relax_MS
Many-state superparamagnetic relaxation: the moment of one particle of spin S
hops between its $2S+1$ projections in a uniaxial anisotropy potential, and the
$\Delta m = \pm 1, 0$ line groups are broadened by the resulting stochastic
process.

Parameters:
- T: effective thickness.
- $\delta$: central shift.
- $\varepsilon$: quadrupole contribution.
- H: saturation hyperfine field, i.e. the field of the fully aligned ($m = S$)
  state. State $m$ sees $H\,m/S$.
- L: Lorentzian width. This model has NO Gaussian width parameter — the
  broadening it produces is the relaxation itself, not a convolution.
- $\theta_k$: axis polar angle.
- $\varphi_h$: axis azimuth.
- A: uniaxial texture order parameter.
- R: relaxation rate — the prefactor of the jump rates
  $\propto R\,[S(S+1)-m(m\mp1)]$. Small R = a static sextet, large R = a
  collapsed (fast-relaxing) line.
- alfa: reduced uniaxial anisotropy barrier (the barrier energy in units of
  $k_BT$). It enters as the Boltzmann factor of the UPWARD jumps only — jumps
  toward lower energy keep a factor 1, which is detailed balance. alfa = 0 is
  free hopping; large alfa freezes the moment near $m = \pm S$.
- S: spin of the particle — the total spin quantum number, giving $2S+1$
  projections. It is the physical size knob of the superparamagnetic particle
  (and therefore also what sets the cost: the model solves a $(2S+1)$-state
  system per velocity point). Structural, so it is never fitted.

### Relax_2S
Two-state (Blume) relaxation: the nucleus jumps between two complete hyperfine
sets. Set H2 = -H1 for a moment that reverses; set the two sets differently for
a jump between two chemical/magnetic states.

Parameters:
- T: effective thickness.
- $\delta_1$, $\varepsilon_1$, H1: state 1 hyperfine set.
- $\delta_2$, $\varepsilon_2$, H2: state 2 hyperfine set.
- L: Lorentzian width.
- $\theta_k$: axis polar angle.
- $\varphi_h$: axis azimuth.
- A: uniaxial texture order parameter.
- Am: magnetic polar-order parameter. Note that a mirror-symmetric pair
  (H2 = -H1 at P1/P2 = 1) is unmagnetised: the Faraday term then cancels for any
  Am, which is correct physics, not a bug.
- $\Omega_{12}$: transition rate between the states, in the same (mm/s) units as
  L, so $\Omega_{12} \ll L$ is slow relaxation and $\Omega_{12} \gg$ the
  splitting is the fast (collapsed) limit.
- P1/P2: population ratio of the two states.

### Hamiltonian
Full Hamiltonian model (magnetic + quadrupole, arbitrary orientation) of a
MOSAIC textured crystal. It replaces the former single-crystal `Hamilton_mc`
(now $A = Am = Ah = 1$) and powder `Hamilton_pc` (now $A = Am = Ah = 0$),
both of which it reproduces exactly, and interpolates continuously between them.

Parameters:
- T: effective thickness.
- $\delta$: central shift.
- Q: quadrupole coupling scale.
- H: magnetic field.
- L: Lorentzian width.
- G: Gaussian width.
- $\eta$: EFG asymmetry parameter.
- $\theta_H$, $\varphi_H$: hyperfine field direction in PAS.
- $\theta$, $\varphi$: PAS direction of the lab reference axis (the radiation
  magnetic field **h** for an SMS source, the beam **k** for a CMS one).
- $\alpha_k$: beam rotation about **h** (SMS only; redundant for CMS).
- A: mosaic order $\langle P_2(\cos\chi)\rangle$, $\chi$ = tilt of the crystal
  direction $(\theta,\varphi)$ away from the lab reference axis. 1 = no spread
  (single crystal), 0 = isotropic tilt, $-1/2$ = tilt confined to $90^\circ$.
- Am: polar order of that same tilt, $S_1 = \langle\cos\chi\rangle =
  Am\sqrt{(1+2A)/3}$. Non-zero only for a magnetised/polar mosaic; it acts
  only through the Faraday (magneto-optical) term, exactly as in the Sextet.
- Ah: order of the crystal azimuth about the reference axis,
  $\langle\cos m\alpha\rangle = Ah^{|m|}$. 1 = $\alpha_k$ sharply defined,
  0 = crystallites uniformly spun about the axis (fiber texture).

Limits: $Ah = 0$ is a fiber texture about the reference axis; at $Q = 0$ and
$Ah = 0$ the model equals the textured Sextet with its axis along that axis and
$A_\mathrm{eff} = A\,P_2(\cos\theta_{BH})$.

#### Reading the geometry in CMS and in SMS

The **same** 15 parameters are used in both source modes; what changes is which
lab axis $(\theta, \varphi)$ points at, and that changes what is measurable.

| | SMS (polarized) | CMS (unpolarized) |
| --- | --- | --- |
| $(\theta, \varphi)$ points at | the radiation magnetic field $\mathbf{h}$ | the BEAM $\mathbf{k}$ (foil normal) |
| the mosaic is a texture about | $\mathbf{h}$ | the beam / foil normal |
| $\alpha_k$ | rotation of the beam about $\mathbf{h}$ — observable | **no effect at all** (an unpolarized beam has no transverse direction to reference); leave it fixed at 0 |
| A | active | active |
| Am | active; its SIGN is observable | active (a thick, off-axis effect), but its SIGN is essentially not |
| Ah | active | still active — a partially ordered azimuth is not the same absorber as a uniformly spun one, even though the reference azimuth $\alpha_k$ itself is unobservable |

So in CMS the sample description is: a mosaic of tilt order A about the foil
normal, with Ah the residual azimuthal order and Am any net magnetisation
along the beam; $(\theta, \varphi)$ is the crystal direction that the beam runs
along, and $\alpha_k$ is a spare parameter that must stay fixed.
Never release $\alpha_k$ in CMS: it is exactly redundant, so the fit
will wander along it without changing $\chi^2$.

### Hamilton_mc, Hamilton_pc (deprecated)
Superseded by `Hamiltonian` and no longer offered in the model dropdown; opening
an old model file rewrites them to `Hamiltonian` with the order parameters above.
They will be removed in a later release.

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
- Num: number of sampling points over one modulation period (rounded internally
  to $6k+1$). The accuracy↔speed knob; structural, so never fitted.
- I13: intensity ratio of the outer (1, 6) to the inner (3, 4) lines, as in
  Sextet.
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

### SCDW
Spin/charge density wave. The spin AXIS is fixed at $(\theta_k, \varphi_h)$;
only the scalar hyperfine parameters (signed field magnitude, central shift,
quadrupole shift) are modulated along the wave, sampled at a fixed number of
phase points over one period and averaged as a thick (polarized) sextet. The
field keeps its SIGN, so a reversed moment automatically swaps the line positions
within the (1,6) and (3,4) pairs. Similar to the SpectrRelax SDW/CDW parameter list.

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
- Am: magnetic polar-order parameter (as in the Sextet model) that scales the
  resolved $\sigma^\pm$ Faraday term.
- $K_\delta H$: central-shift–field correlation (mm/s per field unit).
- $K_\varepsilon H$: quadrupole–field correlation (mm/s per field unit).
- $\Phi$: CDW phase (deg).
- h1, h3, …, h15: eight odd SDW field harmonics (field units); unused ones stay 0.
- d2, d4, d6, d8: four even CDW central-shift harmonics (mm/s); unused ones stay 0.
- N/Γ: grid resolution — the number of grid steps per line width used to bin the
  wave positions. This is the accuracy↔speed knob (larger = finer grid = more
  Voigt evaluations = more accurate = slower; error ~ 1/(N/Γ)). Default 4.
- I13: intensity-ratio control between outer and inner groups.

The wave-phase sampling count (the former ``Num``) is **not** settable: it is
fixed internally at ``models.SDW_NUM = 2000``. Its cost is cheap O(Num)
arithmetic and, because the positions are binned onto the grid, it does not drive
the (Voigt) cost — 2000 is set high enough that the spectrum is converged for any
reasonable wave, so there is nothing to tune. The single accuracy↔speed knob is
therefore the grid resolution N/Γ above.

The resolved $\sigma^\pm$ (Faraday) matrices are kept per site (never the merged
symmetric form). Whether the Faraday term survives the modulation is set by the
WAVE, independently of Am: for a balanced wave (H0 = 0 and
$K_\varepsilon H = K_\delta H = 0$) the $+$/$-$ sites are populated equally and
the Faraday contributions cancel exactly for any Am; a non-zero base field H0
(or field correlation) leaves a real, Am-scaled, thickness-dependent Faraday
signal. With all wave parameters zero and H0 $\ne$ 0 the model reduces to a
single (Sextet-equivalent) sextet at the same Am.

---

## 3. Presets and utility component rows

### Be
Preset impurity component (pre-filled Doublet-like row loaded from Be.txt): the
beamline optics. Edit it in Supp → "Set parameters of Be (optics impurity)".

Parameters are identical to Doublet:
- T, $\delta$, $\varepsilon$, L, G, $\theta_k$, $\varphi_h$, A, G2/G1.

### KB_nano
Preset impurity component (pre-filled Doublet-like row loaded from KB.txt): the
Nanoscope. Edit it in Supp → "Set parameters of KB (Nanoscope impurity)".

Parameters are identical to Doublet:
- T, $\delta$, $\varepsilon$, L, G, $\theta_k$, $\varphi_h$, A, G2/G1.

### Layer
Layer boundary marker for non-commuting thick-matrix propagation.

Parameters:
- none.

Behavior:
- Components above and below Layer are exponentiated as separate layer matrices,
  multiplied in order at the amplitude level. Use it for a genuine stack (two
  foils, a coating on a substrate, differently oriented crystals) — components
  within one layer are averaged before the exponential instead.
- For scalar-only models this behaves like a no-op.

### Library, Load model, Copy, Paste, Insert, Delete
Not models: they act on the table row they were selected from. `Library` opens
the phase library and `Load model` a `.mdl` file browser; both ADD every
component of the chosen model at that row, keeping your current baseline and
re-indexing the added `p[i]` / `=[X,Y]` references (links to the baseline's
first 8 parameters stay as they are). `Copy`/`Paste` move a row (model, values
and fix states) through an internal clipboard; `Insert` and `Delete` add/remove
a row and renumber every `p[i]` and `=[X,Y]` reference for you.

---

## 4. Expression/distribution helper rows

`Distr`, `Corr` and `Recon` attach to the physical model row ABOVE them; their
`par` is the parameter's column number WITHIN that row (1 = its second parameter,
i.e. $\delta$ for most models). The parameter `par`
selects is framed in grey and becomes read-only, so the choice can be checked at
a glance. `Expression` and `Variables` stand on their own.

### Distr
Adds a parameter distribution to the preceding physical model. Several `Distr`
rows on the same component build a multidimensional distribution.

Parameters:
- par: which parameter of the model above is distributed.
- L: left bound of distribution axis.
- R: right bound of distribution axis.
- Num: number of grid points.
- Probability density function: expression in variable X (the axis), e.g.
  `exp(-X**2/2)`.

### Corr
Makes a second parameter follow the distributed one, so the distribution becomes
correlated. May only be placed under a `Distr`/`Corr`/`Recon`.

Parameters:
- par: which parameter of the model above is driven.
- Dependency function: expression in variable X (the current value of the
  distributed parameter), e.g. `0.3*X`.

### Recon
Distribution RECONSTRUCTION: same axis definition as `Distr`, but the shape is
fitted instead of given. The Num weights are free fit variables held beside the
parameter array (so parameter numbering never shifts with Num), regularized by
two Tikhonov terms.

Parameters:
- par, L, R, Num: as for `Distr`.
- D_dif: weight of the first-derivative smoothness penalty, 0…1 (0 = free).
- D_dif2: the same for the second derivative.
- weights: the reconstructed distribution as a comma-separated list. Leave it
  empty for a uniform start — the fit writes the result back here.

Raise D_dif / D_dif2 if the reconstruction oscillates; the two penalties are the
only thing keeping an under-determined shape stable.

### Expression
A free algebraic expression evaluated on the parameter array p. The result is a
parameter in its own right — point real parameters at it with `=[X,Y]` (see
*Supp -> Help (hidden features)*).

Parameters:
- Expression: formula text, for example `p[0]` or `p[5]*0.5`.

Expressions are calculated in list order - thus, expression should not refer to another expression which stays below (but it can refer to any parameter)

Avoid chain links in expression - if some parameter number `Z` is linked like `=[X,Y]` then expressions should not use `p[Z]` but `p[X]`.

### Variables
Helper row of scalar placeholders to reference from expressions and links; the
row is as wide as the table (currently V1 … V27).

Parameters:
- V1 … V27: user-defined scalar values.

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
- A controls second-moment alignment ($A=0$ random powder, $A=1$ aligned along
  the axis, $A=-1/2$ the axis confined to the plane perpendicular to it). Every
  model reduces EXACTLY to its former scalar (powder) form at $A=0$ — which is
  why $A=0$ is the default everywhere.
- Am controls first-moment magnetic polar order in the Faraday-active models
  (Sextet, MDGD, Relax_2S, SCDW, Hamiltonian), bounded by
  $S_1 = Am\sqrt{(1+2A)/3}$.
- Ah (Hamiltonian only) controls the azimuthal order about the reference axis.

Am is a purely THICK, off-axis observable: it enters only through the
off-diagonal Faraday term, so it does nothing in the thin limit and nothing when
the axis lies in the polarization plane ($\theta_k = 90°$).

Practical fitting advice:
- Start with A = 0 (powder-like) unless strong texture is expected.
- Keep orientation angles fixed initially, then release if residuals suggest anisotropy.
- Release Am only when data quality supports it, and only for a genuinely
  magnetised sample — a mirror-symmetric (unmagnetised) system cancels the
  Faraday term whatever Am says.

---

## 6. Source mode notes

- SMS mode: linearly polarized readout. The polarization degree is settable in
  *Supp -> Set polarization* (default 0.98).
- CMS mode: unpolarized half-trace readout.

