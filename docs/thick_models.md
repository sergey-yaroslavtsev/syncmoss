# Polarized ("thick") Mössbauer models — equations

This document defines every equation and **explicit cross-section matrix** used
by the polarized **"(thick)"** model variants and the `Layer` marker implemented
in `syncmoss/models.py` (function `TImod` and its helpers). The code follows it
line by line.

Motivation: for a linearly polarized synchrotron (SMS) source the scalar
transmission integral $T=\exp(-\text{absorber})$ is wrong, because a resolved
circular/linear absorption line cannot attenuate a linearly polarized beam by
more than 50 %. The correct object is a **2×2 cross-section matrix** acting in the
polarization plane.

---

## 1. Framework

### 1.1 Geometry (lab frame)

* Beam direction $\mathbf{k}=\hat z$.
* Radiation magnetic field (the M1 "polarization") $\mathbf{h}=\mathbf{e}_1=\hat x$.
* Second polarization $\mathbf{e}_2=\hat y=\mathbf{k}\times\mathbf{h}$.

A component's symmetry axis ($\mathbf V_{zz}$ for a doublet, $\mathbf B_{\mathrm{hf}}$
for a sextet) is given by $(\theta_h,\varphi_h)$ — **$\theta_h$ from the beam
$\mathbf k$, $\varphi_h$ from the polarization $\mathbf h$**:

$$
\hat{\mathbf n}=\begin{pmatrix}n_x\\ n_y\\ n_z\end{pmatrix}
=\begin{pmatrix}\sin\theta_h\cos\varphi_h\\ \sin\theta_h\sin\varphi_h\\ \cos\theta_h\end{pmatrix},
\qquad
\cos\beta=\mathbf h\!\cdot\!\hat{\mathbf n}=n_x=\sin\theta_h\cos\varphi_h ,
$$

with $\beta$ the angle between $\mathbf h$ and the axis. (`_axis_xyz` returns
$(n_x,n_y,n_z)$.)

### 1.2 Transmission

A single homogeneous layer with total 2×2 cross-section $\hat\Sigma(v)$ transmits,
for a fully linearly polarized incident beam $\mathbf e_1=\mathbf h$,

$$
\boxed{\,T(v)=\big[\exp(-\hat\Sigma(v))\big]_{11}\,}.
$$

This is the fully-polarized ($p=1$) special case; the general readout for a
partially polarized SMS beam or an unpolarized CMS source is in §4.

Each thick component adds its matrix to the layer accumulator; the matrices are
**summed before the single matrix exponential** (homogeneous mixture). Scalar /
powder components are isotropic in the polarization plane ($\propto\mathbb I_2$)
and factor out as an ordinary scalar:

$$
T(v)=\exp\!\Big(-\!\!\sum_{\text{scalar }j}\!A_j(v)\Big)\,
\Big[\exp\!\big(-\!\!\sum_{\text{thick }i}\!\hat\Sigma_i(v)\big)\Big]_{11}.
$$

The overall prefactor equals the scalar model's,
$K=\pi\,\dfrac{\Gamma_{\mathrm{nat}}}{2E_0}\,c\,\mu$ (`Kpref`; $\mu=$ `MulCo`).

### 1.3 Layers (`Layer` marker — non-commuting product)

The polarization leaving one physical layer enters the next, and layer matrices
**do not commute**. For components $A,B$ (layer 1), a `Layer` marker, then $C$
(layer 2):

$$
\boxed{\,T(v)=\Big[\exp(-\hat\Sigma_C)\,\exp(-\hat\Sigma_{A+B})\Big]_{11}\,}
\ne\Big[\exp(-\hat\Sigma_{A+B+C})\Big]_{11}.
$$

`Layer` has no parameters; for pure scalars it is a no-op.

### 1.4 Matrix exponential (closed form, `_expm_neg`)

For any 2×2 $M$ with $s=\tfrac12\operatorname{tr}M$, $q=\sqrt{s^2-\det M}$:

$$
\exp(M)=e^{s}\!\left[\cosh q\,\mathbb I_2+\frac{\sinh q}{q}\,(M-s\,\mathbb I_2)\right],
\qquad \frac{\sinh q}{q}\to1\ (q\to0).
$$

### 1.5 Powder normalisation & the 50 % limit

Using $\langle \mathsf P_\perp\rangle_{\text{sphere}}=\tfrac13\mathbb I_2$ and
$\langle n_z\rangle=0$, **every matrix below averages over the sphere to
$\mathbb I_2$**, so a thick component reduces *exactly* to its scalar twin in the
powder limit (verified $<10^{-5}$). A single $\sigma$ line with the axis along the
beam ($\theta_h=0$) has $\hat M_{\sigma^\pm}=\tfrac32(\mathbb I_2\pm\mathrm iJ)$
with eigenvalues $\{0,3\}$ — the zero eigenvalue is the transparent wrong-handed
component, giving the $T\to\tfrac12$ saturation floor.

### 1.6 Explicit building-block matrices

With $\;\mathbb I_2=\begin{pmatrix}1&0\\0&1\end{pmatrix}$,
$\;J=\begin{pmatrix}0&1\\-1&0\end{pmatrix}$, and the in-plane projector

$$
\mathsf P_\perp=\mathbf n_\perp\mathbf n_\perp^{\mathsf T}=
\begin{pmatrix} n_x^2 & n_xn_y\\ n_xn_y & n_y^2\end{pmatrix}
=\begin{pmatrix}\sin^2\theta_h\cos^2\varphi_h & \sin^2\theta_h\cos\varphi_h\sin\varphi_h\\[2pt]
\sin^2\theta_h\cos\varphi_h\sin\varphi_h & \sin^2\theta_h\sin^2\varphi_h\end{pmatrix},
$$

the **four reusable cross-section matrices** are:

$$
\textbf{Doublet line A / }\Delta m=\pm1:\quad
\hat M_A=\tfrac32(\mathbb I_2-\mathsf P_\perp)=
\frac32\begin{pmatrix} 1-n_x^2 & -n_xn_y\\ -n_xn_y & 1-n_y^2\end{pmatrix},
$$

$$
\textbf{Doublet line B (mixed):}\quad
\hat M_B=\tfrac12\mathbb I_2+\tfrac32\mathsf P_\perp=
\begin{pmatrix} \tfrac12+\tfrac32 n_x^2 & \tfrac32 n_xn_y\\[2pt] \tfrac32 n_xn_y & \tfrac12+\tfrac32 n_y^2\end{pmatrix},
$$

$$
\boldsymbol{\sigma^{\pm}}\ (\Delta m=\pm1):\quad
\hat M_{\sigma^{\pm}}=\tfrac32\big[(\mathbb I_2-\mathsf P_\perp)\pm\mathrm i\,n_z J\big]=
\frac32\begin{pmatrix} 1-n_x^2 & -n_xn_y\pm\mathrm i\,n_z\\[2pt] -n_xn_y\mp\mathrm i\,n_z & 1-n_y^2\end{pmatrix},
$$

$$
\boldsymbol{\pi}\ (\Delta m=0):\quad
\hat M_{\pi}=3\,\mathsf P_\perp=
\begin{pmatrix} 3n_x^2 & 3n_xn_y\\ 3n_xn_y & 3n_y^2\end{pmatrix},
$$

$$
\textbf{Faraday-averaged }\sigma:\quad
\hat M_{\sigma}^{\mathrm{sym}}=\tfrac32(\mathbb I_2-\mathsf P_\perp)=\hat M_A
=\frac32\begin{pmatrix} 1-n_x^2 & -n_xn_y\\ -n_xn_y & 1-n_y^2\end{pmatrix}.
$$

($n_x=\sin\theta_h\cos\varphi_h,\ n_y=\sin\theta_h\sin\varphi_h,\ n_z=\cos\theta_h$.)
Powder averages: $\langle\hat M_A\rangle=\langle\hat M_B\rangle=\langle\hat M_{\sigma^\pm}\rangle
=\langle\hat M_\pi\rangle=\langle\hat M_\sigma^{\mathrm{sym}}\rangle=\mathbb I_2$.

There is **no Singlet thick model**: a singlet is isotropic, $\hat M=\mathbb I_2$,
so its thick form equals the scalar singlet exactly.

---

## 2. Cross-section per model

In every case the per-line scalar coefficients are the **scalar model's
intensities with the asymmetry $A$ fixed to its isotropic value $A=\tfrac12$**;
the orientation dependence is carried entirely by the matrices above. Line
positions, widths and lineshapes are identical to the scalar twin. $T_a$ denotes
the effective thickness.

### 2.1 Doublet `Doublet_(thick)` (9 params: $T,\delta,\varepsilon,\Gamma_L,\Gamma_{G},\theta_h,\varphi_h,A,\Gamma_{G2}/\Gamma_{G1}$)

Line A ($\Delta m=\pm1$, at $\delta+\varepsilon$) uses $\hat M_A$; line B (mixed,
at $\delta-\varepsilon$) uses $\hat M_B$:

$$
\hat\Sigma(v)=K\,T\,\tfrac{T_a}{2}\Big[V_A(v-\delta-\varepsilon)\,\hat M_A+V_B(v-\delta+\varepsilon)\,\hat M_B\Big].
$$

Thin-limit $(1,1)$ element: $(\hat M_A)_{11}=\tfrac32\sin^2\beta\Rightarrow f_+=\tfrac34\sin^2\beta$;
$(\hat M_B)_{11}=\tfrac12+\tfrac32\cos^2\beta\Rightarrow f_-=\tfrac14(1+3\cos^2\beta)$.

### 2.2 Sextet `Sextet_(thick)` (14 params: $T,\delta,\varepsilon,H,\Gamma_L,\Gamma_G,\theta_h,\varphi_h,A,A_m,a_+,a_-,\Gamma_H,I_1/I_3$)

Lines $1{=}\sigma^-,2{=}\pi,3{=}\sigma^+,4{=}\sigma^-,5{=}\pi,6{=}\sigma^+$. With
$r\equiv I_1/I_3$ and $A=\tfrac12$:

$$
I_1=I_6=\frac{T_a\,r}{3(r+1)},\quad I_2=I_5=\frac{T_a}{6},\quad I_3=I_4=\frac{T_a}{3(r+1)},
$$
$$
\hat\Sigma(v)=K\Big[I_1(V_1\hat M_{\sigma^-}+V_6\hat M_{\sigma^+})
+I_2(V_2+V_5)\hat M_\pi+I_3(V_3\hat M_{\sigma^+}+V_4\hat M_{\sigma^-})\Big].
$$

Positions $v_{1..6}$ and widths $\Gamma_{16},\Gamma_{25},\Gamma_{34}$ (from
$\Gamma_G,\Gamma_H$) and shifts $a_\pm$ are exactly the scalar `Sextet`.

### 2.3 MDGD `MDGD_(thick)` (17 params)

Same matrices and $A=\tfrac12$ intensities as the sextet; only the per-line
Gaussian widths differ — they use the MDGD correlated-distribution widths
$\Gamma^{(j)}_{\text{final}}(\Gamma_{\text{uni}},\Gamma_H,D_{\delta\varepsilon},
D_{\delta H},D_{\varepsilon H})$, exactly as scalar `MDGD`.

### 2.4 Two-state relaxation `Relax_2S_(thick)` (14 params)

Each line is a separate Blume two-state lineshape $B_{m_0\to m_1}(v)$, so each keeps
its own $\sigma^\pm/\pi$ matrix (Faraday term included). With $A=\tfrac12$
($I_1=\tfrac14T_a,I_2=\tfrac16T_a,I_3=\tfrac1{12}T_a$):

$$
\hat\Sigma(v)=K\Big[
I_1\big(B_{-\frac12\to-\frac32}\hat M_{\sigma^-}+B_{\frac12\to\frac32}\hat M_{\sigma^+}\big)
+I_2\big(B_{-\frac12\to-\frac12}+B_{\frac12\to\frac12}\big)\hat M_\pi
+I_3\big(B_{-\frac12\to\frac12}\hat M_{\sigma^+}+B_{\frac12\to-\frac12}\hat M_{\sigma^-}\big)\Big].
$$

### 2.5 Many-state relaxation `Relax_MS_(thick)` (11 params) — *non-trivial*

The relaxation solver (`_relax_MS_groups`, shared with scalar `relax_MS`) returns
three **group** spectra that already merge the two lines of each $\Delta m$ family
by spin state: $g_1$ (outer $\Delta m=\pm1$), $g_2$ (middle $\Delta m=0$), $g_3$
(inner $\Delta m=\pm1$). Because each group blends $\sigma^+$ and $\sigma^-$, the
$\pm\mathrm i\,n_zJ$ Faraday terms cancel and the **symmetric** matrix
$\hat M_\sigma^{\mathrm{sym}}$ is used:

$$
\hat\Sigma(v)=K\,T\Big[g_1(v)\,\hat M_\sigma^{\mathrm{sym}}+g_2(v)\,\hat M_\pi+g_3(v)\,\hat M_\sigma^{\mathrm{sym}}\Big],
$$

where the $g_i$ already carry the isotropic $3:2:1$ weights and the thickness.
$\hat M_\sigma^{\mathrm{sym}}$ still has eigenvalues $\{0,3\}$ for an axis in the
polarization plane, so the 50 % limit holds; only the (unresolved) longitudinal
circular dichroism is dropped.

### 2.6 Single-crystal Hamiltonian `Hamilton_mc_(thick)` (12 params)

The shared core `_ham_mono_core` diagonalises the combined quadrupole + magnetic
Hamiltonian and returns, per transition $k=1..8$, the polarization-independent
spherical amplitudes $g_0,g_1,g_2$ (for $q=+1,0,-1$, including the
$\tfrac14\sqrt{\cdot/\pi}$ prefactors) and the positions $S_k$. The polarization
basis in the PAS uses $(\theta_h,\varphi_h)$ and the **beam-rotation angle
$\alpha_k$**:

$$
\hat{\boldsymbol\theta}=(\cos\theta_h\cos\varphi_h,\cos\theta_h\sin\varphi_h,-\sin\theta_h),\quad
\hat{\boldsymbol\varphi}=(-\sin\varphi_h,\cos\varphi_h,0),
$$
$$
\mathbf k=\cos\alpha_k\,\hat{\boldsymbol\theta}+\sin\alpha_k\,\hat{\boldsymbol\varphi},\qquad
\mathbf e_2=\mathbf k\times\mathbf h=\sin\alpha_k\,\hat{\boldsymbol\theta}-\cos\alpha_k\,\hat{\boldsymbol\varphi}.
$$

For a polarization at spherical angles $(\theta_p,\varphi_p)$,
$F_{+}=\sqrt2\sin\theta_p(-\mathrm i)e^{\mathrm i\varphi_p}$,
$F_{0}=\sqrt2\cos\theta_p(\mathrm i)$,
$F_{-}=\sqrt2\sin\theta_p(\mathrm i)e^{-\mathrm i\varphi_p}$, and
$A^{(p)}_k=g_0^k F_{+}+g_1^k F_{0}+g_2^k F_{-}$. With $A_1=A^{(\mathbf e_1)}$,
$A_2=A^{(\mathbf e_2)}$, the **per-transition cross-section matrix** (explicit) is

$$
\hat P_k=\pi\begin{pmatrix}\,|A_1^k|^2 & A_1^k\,\overline{A_2^k}\,\\[3pt]
\,\overline{A_1^k}\,A_2^k & |A_2^k|^2\,\end{pmatrix},
\qquad
\hat\Sigma(v)=K\,T_a\sum_{k=1}^{8} V(v-S_k)\,\hat P_k .
$$

By construction $(\hat P_k)_{11}=\pi|A_1^k|^2$ equals the scalar `Ham_mono`
intensity, so the thin limit reproduces the scalar SMS model.

**The two sources use different geometries** for this single-crystal model — the
angles $(\theta,\varphi)$ (columns `tetr, phir`) are reinterpreted, and the code
selects one of two cores by `Met`:

* **SMS** ($\texttt{Met}\ne1$, linearly polarized; core `Ham_mono_thick`) — the
  construction above: $(\theta_h,\varphi_h)$ is the radiation field $\mathbf h$ in
  the PAS and $\alpha_k$ rotates the beam $\mathbf k=\mathbf e_1\times\mathbf e_2$
  about $\mathbf h$. For a thick sample $\alpha_k$ is a **genuine observable** — it
  mixes the two polarization channels during propagation (the off-diagonal
  $\hat P_{k,01}\propto A_1\overline{A_2}$), so it affects even the $(1,1)$ readout.

* **CMS** ($\texttt{Met}=1$, unpolarized; core `Ham_mono_thick_CMS`) —
  $(\theta,\varphi)$ is instead the **beam direction $\mathbf k$** in the PAS, and
  the two polarizations $\mathbf e_1=\hat{\boldsymbol\theta}(\mathbf k)$,
  $\mathbf e_2=\hat{\boldsymbol\varphi}(\mathbf k)$ are the spherical-basis unit
  vectors perpendicular to $\mathbf k$. The readout (per §4) is the half-trace
  $\tfrac12\operatorname{tr}\hat P_k=\tfrac{\pi}{2}\big(|A_1^k|^2+|A_2^k|^2\big)
  =\tfrac{\pi}{2}\lVert\mathbf d_{k,\perp}\rVert^2$, which is **invariant under any
  rotation of $(\mathbf e_1,\mathbf e_2)$ about $\mathbf k$**. There is therefore
  **no $\alpha_k$**: it would only spin that arbitrary transverse basis, so it is
  redundant (a true no-op) and the user leaves it fixed. The thin limit
  $\tfrac12\operatorname{tr}\hat P_k$ equals the scalar `Ham_mono_CMS` intensity for
  the same beam $(\theta,\varphi)$ (verified $<10^{-10}$).

### 2.7 Anharmonic spin modulation `ASM_(thick)` (14 params) — *non-trivial, with a geometric assumption*

The hyperfine field direction **rotates along the cycloid** while $\mathbf h$ is
fixed (a distribution of $H$ orientations, not of $\mathbf h$). The modulation
core `ASM_thick_terms` returns, at $N$ points $x_j$ of one period:
$co_j=\operatorname{sn}^2(x_j,m)=\cos^2\vartheta_j$ (the local moment polar angle
relative to the cycloid axis; $(3co-1)/2$ modulates $H$ and $\varepsilon$ as in
scalar `ASM`) and the six line positions $v_{1..6}(x_j)$.

**Geometry (assumption, documented):** $(\theta_h,\varphi_h)$ is the cycloid
(anharmonicity) axis $\hat{\mathbf n}$ in the lab frame; the moment swings in the
plane spanned by $\hat{\mathbf n}$ and the in-plane part of $\mathbf h$,

$$
\hat{\mathbf e}=\frac{\mathbf h-(\mathbf h\!\cdot\!\hat{\mathbf n})\hat{\mathbf n}}{\lVert\cdot\rVert},
\qquad
\hat{\mathbf m}_j=\cos\vartheta_j\,\hat{\mathbf n}+\sin\vartheta_j\,\hat{\mathbf e},
\quad\cos\vartheta_j=\sqrt{co_j}.
$$

Because the cycloid samples both senses of the moment, the Faraday term averages
out and the symmetric matrices $\hat M_\sigma^{\mathrm{sym}},\hat M_\pi$ — built
from the in-plane components of $\hat{\mathbf m}_j$ — are used. The cross-section
is the modulation average

$$
\hat\Sigma(v)=\frac{K}{N}\sum_{j=1}^{N}\Big[
I_1(V_{1,j}+V_{6,j})\hat M_\sigma^{\mathrm{sym}}(\hat{\mathbf m}_j)
+I_2(V_{2,j}+V_{5,j})\hat M_\pi(\hat{\mathbf m}_j)
+I_3(V_{3,j}+V_{4,j})\hat M_\sigma^{\mathrm{sym}}(\hat{\mathbf m}_j)\Big],
$$

with $V_{k,j}=V(v-v_{k,j})$ and the isotropic ($A=\tfrac12$) weights $I_{1,2,3}$,
$r=I_1/I_3$ as in the sextet. **Caveat:** the cycloid plane is taken to contain
$\mathbf h$; a true powder of cycloids would also average over the plane
orientation. Confirm/refine for quantitative work.

---

## 3. Parameter mapping (scalar → thick)

| model | thick change | count |
|---|---|---|
| Doublet | $A_{\rm asym}\to\theta_h,\varphi_h,A$ | 7→9 |
| Sextet | $A_{\rm asym}\to\theta_h,\varphi_h,A$; $+A_m$ | 11→14 |
| MDGD | $A_{\rm asym}\to\theta_h,\varphi_h,A$; $+A_m$ | 14→17 |
| Relax_MS | $A_{\rm asym}\to\theta_h,\varphi_h,A$ | 9→11 |
| Relax_2S | $A_{\rm asym}\to\theta_h,\varphi_h,A$; $+A_m$ | 11→14 |
| Hamilton_mc | $+\,\alpha_k$ (no texture) | 11→12 |
| ASM | $A_{\rm asym}\to\theta_h,\varphi_h,A$ | 12→14 |
| Layer | marker, no params | 0 |

(The Faraday-active models — Sextet, MDGD, Relax_2S — carry one further parameter
$A_m$, the magnetic polar-order fraction, immediately after $A$; see §3.2.)

(Singlet has no thick form. For Hamilton the existing $(\theta,\varphi)$ already are
the SMS radiation direction $(\theta_h,\varphi_h)$ — under CMS they are the beam
direction $\mathbf k$ instead, see §2.6; only $\alpha_k$ is added — see §3.1 for why
Hamilton gets no texture parameter.)

### 3.1 Uniaxial (fiber) texture parameter $A$

Immediately after the orientation angles $(\theta_h,\varphi_h)$ each matrix-based
thick model carries a single **uniaxial (fiber) texture order parameter**
$A\in[-\tfrac12,1]$ (locked by default, like the angles). A textured powder whose
symmetry axes follow an axially symmetric orientation distribution about the axis
$(\theta_h,\varphi_h)$ averages — *before* the matrix exponential — to

$$
\boxed{\;\langle\hat M\rangle = (1-A)\,\mathbb I_2 + A\,\hat M(\theta_h,\varphi_h)\;}
$$

applied to **every** building block ($\hat M_A,\hat M_B,\hat M_{\sigma^\pm},
\hat M_\pi,\hat M_\sigma^{\rm sym}$; helper `_texture_blend`). Since every block
powder-averages to $\mathbb I_2$, this interpolates continuously:

* $A=0$ → random powder: $\langle\hat M\rangle=\mathbb I_2$, the **scalar model
  recovered exactly at any thickness**;
* $A=1$ → perfect alignment: the single-crystal thick model at
  $(\theta_h,\varphi_h)$ — a **bit-exact no-op** on the previous behaviour;
* $A=-\tfrac12$ → perfect planar texture (axes uniformly $\perp$ the axis).

$A$ is the order parameter $\langle P_2(\cos\psi)\rangle$ ($\psi$ = angle to the
texture axis). The blend $\langle\hat M\rangle=(1-A)\Iop+A\,\hat M$ is applied only
to the **quadratic** building blocks ($\hat M_A,\hat M_B,\hat M_\pi,\hat
M_\sigma^{\rm sym}$; helper `_texture_blend`). The magneto-optical (Faraday)
$\pm\mathrm i n_z J$ term of a resolved $\sigma^\pm$ line is **linear** in the axis
and averages to a **separate** first moment, the polar-order parameter $S_1$,
carried by its own fit parameter $A_m$ — see §3.2. With $A_m=0$ (the default) the
$\sigma^\pm$ blocks use the Faraday-averaged $\hat M_\sigma^{\rm sym}$, so $A=0$
still recovers the powder exactly and $A=1$ an *unmagnetized* aligned single
crystal; $A=A_m=1$ recovers the fully magnetized single-crystal $\sigma^\pm$ lines.
Nothing else changes: the matrices are still summed and a single $2\times2$
exponential is taken.

### 3.2 Magnetic polar-order parameter $A_m$ ($S_1$)

The three **Faraday-active** models — Sextet, MDGD, Relax_2S, the ones whose
$\sigma^+$ and $\sigma^-$ partners are resolved at different energies — carry,
immediately after $A$, a second texture moment. While $A=\langle
P_2(\cos\chi)\rangle$ measures *alignment*, the Faraday term averages to the first
moment $S_1=\langle\cos\chi\rangle$, the net **polar** (magnetic) order along the
texture axis. The textured $\sigma^\pm$ blocks are (helper `_texture_s1`, applied
in the model on top of the symmetric blend):

$$
\boxed{\;\langle\hat M_{\sigma^\pm}\rangle=(1-A)\,\Iop+A\,\tfrac32(\Iop-\Pperp)
\;\pm\;\tfrac32\,\mathrm i\,S_1\,n_z\,J\;},
\qquad n_z=\cos\theta_h .
$$

The two moments are not independent: Cauchy–Schwarz gives $S_1^{\,2}\le\tfrac13(1+2A)$.
We therefore parametrise $S_1$ by the fit parameter $A_m\in[-1,1]$, its fraction of
that bound,

$$
S_1=A_m\sqrt{\tfrac{1+2A}{3}},
$$

so the fit can never leave the physical region whatever $A$ is (at the planar limit
$A=-\tfrac12$ the bound is $0$, forcing $S_1=0$). $S_1$ is non-zero only for a
**magnetized** texture (the $+\Bhf$/$-\Bhf$ domains unequally populated), so:

* **$A_m=0$ (default)** — unmagnetized: the $\sigma^\pm$ blocks reduce to the
  Faraday-averaged $\hat M_\sigma^{\rm sym}$; this is the whole texture model for
  every EFG (quadrupole) texture and every unmagnetized magnetic texture, at any
  alignment $A$. It also restores the exact random-powder limit at $A=0$.
* **$A_m=A=1$** — the fully magnetized single crystal, reproducing the
  single-crystal $\hat M_{\sigma^\pm}$ of §1.6 exactly (a bit-exact match).

Two caveats, both consequences of the readout being real (§4): $A_m$ is a **purely
thick, off-axis observable** — in the thin limit the Faraday term is off-diagonal
and traceless, so $S_1$ affects neither thin polarized nor thin unpolarized spectra
(like the layer-order effect); and only $|S_1|$ is measurable from **one** layer —
reversing $A_m$ conjugates $\hat\Sigma$, and the transmission (the real $(1,1)$
element for SMS, the real half-trace for CMS) is invariant under conjugation. The
sign of $A_m$ is observable only *relatively*, between two Faraday-active
components mixed in one absorber or between stacked layers. The models that already
use the Faraday-averaged $\hat M_\sigma^{\rm sym}$ — Relax_MS (grouped $\sigma$),
ASM (cycloid average) — and the doublet (no $\sigma^\pm$ splitting) never need
$A_m$ and do not carry it.

**Hamilton_mc_(thick) has no texture parameter.** Its per-transition matrices
$\hat P_k$ depend on the *full* crystallite orientation (the anisotropic-EFG
$\eta\ne0$ couples the EFG eigenframe to the texture frame), so a genuine fiber
average needs the closed-form rank-$\le2$ (Wigner-$D$, $\ell\le2$) average — more
than one parameter. A single blend would be a two-phase "aligned + random"
mixture, not a fiber texture, so it is deliberately omitted.

---

## 4. Source readout: incident-beam polarization

Sections 1–3 build the absorber cross-section $\hat\Sigma(v)$; it is a property of
the *sample* and is **independent of the source**. What the source fixes is only how
the transmission is *read out* of $\hat\Sigma$, through the $2\times2$ polarization
density matrix $\rho$ of the incident beam:

$$
C_{\mathrm a}(v)=\operatorname{tr}\!\big[\exp(-\hat\Sigma(v))\,\rho\big],
\qquad
\rho=\operatorname{diag}\!\Big(\tfrac{1+p}{2},\tfrac{1-p}{2}\Big)\ \text{(linear pol. degree }p\text{ along }\mathbf h),
$$

so $C_{\mathrm a}=\tfrac{1+p}{2}\,[\exp(-\hat\Sigma)]_{11}+\tfrac{1-p}{2}\,[\exp(-\hat\Sigma)]_{22}$.
With `Layer`s (§1.3) the same trace is taken of the layer product $\hat{\mathcal T}$,
which is why $\operatorname{tr}$ is cyclic and an unpolarized source cannot tell the
stacking order apart. Both source modes are supported:

* **SMS (synchrotron)** — linearly polarized. The degree of polarization is the
  `pol` argument of `TI` (default `0.98`, a realistic SMS beam), which forwards it
  to `TImod` as `sms_pol`:
  * `p = 1.0` → fully polarized, reads the $(1,1)$ element only — the original
    behaviour.
  * `p ≈ 0.98` → a realistic SMS beam (the default); `p = 0.0` → unpolarized.

  It is editable at runtime from the GUI (**Supp → "Set polarization"**); the value
  is stored on the app as `SMS_pol` and passed into every `TI` call.

* **CMS (conventional radioactive source, `Met == 1`)** — unpolarized, $\rho=\tfrac12 I$,
  so $C_{\mathrm a}=\tfrac12\operatorname{tr}\exp(-\hat\Sigma)=\tfrac12\big(e^{-\lambda_+}+e^{-\lambda_-}\big)$,
  the average of the two eigen-channel transmissions. This 1:1 mixture is **fixed**
  (an unpolarized beam defines no direction in the polarization plane) and does **not**
  read `pol`/`sms_pol`. Thick models are therefore **valid for a CMS source**
  and are *no longer refused*.

Consequences for a CMS spectrum (the user is responsible for these): for the
matrix-based models (Doublet/Sextet/MDGD/Relax/ASM) the half-trace depends only on
the polar angle $\theta_h$, so the azimuth $\varphi_h$ has **no effect** (an
unpolarized source defines no direction in the polarization plane); a resolved
allowed line saturates at exactly $50\%$ absorption for any orientation, a forbidden
line vanishes, and a random powder shows no floor. (`Hamilton_mc_(thick)` is the one
exception — under CMS its $(\theta,\varphi)$ are the **beam direction $\mathbf k$ in
the crystal frame**, so *both* angles are physical (they aim the beam through the
anisotropic crystal), whereas $\alpha_k$ becomes redundant; see §2.6.) Because
the matrix exponential is convex, $\tfrac12(e^{-\lambda_+}+e^{-\lambda_-})\ne
e^{-\frac12(\lambda_++\lambda_-)}$, so the matrix treatment (not a scalar effective
thickness) is still required for a thick oriented absorber even with an unpolarized
source. In the GUI these orientation angles are **locked by default** for every thick
model (untick their *fix* box to refine them).

---

## 5. Equivalence with thin models (validation)

In the thin limit ($T_a\to0$), a thick model at orientation $(\theta_h,\varphi_h)$
equals the corresponding **thin** model whose asymmetry is

$$
\boxed{A=\frac{2\cos^2\beta}{1+\cos^2\beta}},\qquad \cos\beta=\sin\theta_h\cos\varphi_h,
$$

for Doublet, Sextet, MDGD, Relax_MS and Relax_2S (with $I_1/I_3$ unchanged). For
`Hamilton_mc` the thin equal is `Hamilton_mc` itself with the same
$(\theta,\varphi)$ (any $\alpha_k$). `ASM_(thick)` has no single-$A$ thin equal
because the orientation varies per cycloid point. See
`tests/` / `docs/thick_vs_thin.md` for the numerical comparison.
