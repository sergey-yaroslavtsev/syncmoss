# SYNCmoss help (extended)

The full text behind *Supp -> Help*, which keeps only the short reminders: the
sections have the same numbers in both. Section 13 holds the longer notes on the
models; their parameter lists are in *Supp -> Models description*. Type a word
into the **Find** box above (**Ctrl+F**) to jump to it.

---

## 1. Simulation without a spectrum

Type `Model_<range>` into the spectrum-path box instead of a file name and press
**Show model**. The model is calculated on a synthetic grid of 4096 points over
±`range` mm/s, with no experimental data.

- `Model_6` -> −6 … +6 mm/s. `Model_6.5` works too; the keyword is
  case-insensitive (`model_6`).
- The spelling must be exact. Anything else in the box is read as a spectrum
  path, so a typo such as `models_6` is reported as "not a spectrum file" — the
  message reminds you of the keyword.
- **Show spectrum** and **Fit** have nothing to work on in this mode and say so.
- With `Nbaseline` rows in the model, each section gets its own synthetic grid.

## 2. Linking parameters

Type `=[X,Y]` into a parameter **value** field instead of a number. That
parameter then follows another one:

    value = p[X] * Y

`X` is the flat parameter number, `Y` a multiplier. So `=[12,1]` means "equal to
parameter 12", `=[12,0.5]` means "half of parameter 12", `=[12,-1]` means
"minus parameter 12".

- A finished link is shown on a **darkorange** field, so a parameter that
  follows another one stands apart from the fitted ones. A half-typed link
  (`=[,1]`) stays plain until both numbers are in.
- The right-click menu of a value field offers *link to another parameter*: it
  puts in `=[,1]` with the cursor where the number goes.
- Links are kept in step when you insert or delete model rows: the referenced
  numbers are renumbered for you. The same holds for every `p[i]` inside an
  `Expression`, `Distr` or `Corr` text — which is why a parameter there is
  written exactly `p[12]`: no spaces, a plain number. Show model and Fit
  refuse any other spelling (`p [12]`, `p[ 12 ]`, `p[10+2]`, `p[-1]`).
- If the parameter a link pointed at is itself deleted, the field is emptied and
  turned **red**. Show model and Fit then refuse to start until you fill it in.
- The link is applied both when showing the model and inside every fit
  iteration, so the linked parameter never has its own degree of freedom.
- A link is also how a parameter is shared between components — e.g. one common
  central shift for two sextets.

**Independent values.** Type `=(X)` instead of a number to make a parameter
*independent*: in a simultaneous one-model fit (section 7.6) every spectrum gets a
value of its own, all of them starting from X — every other free parameter has
one value shared by all the spectra. Anywhere else it is simply the number X (so
the model can be tried on one spectrum first). It is shown on a **lightgreen**
field; `=()` without its X stays plain and blocks **Show model** and **Fit**,
and X must lie inside the parameter's bounds. With several spectra in the path
box and no `Nbaseline` row, the right-click menu of a value field offers *make
it independent*, which puts in `=()` with the cursor between the brackets. An
`=(X)` with its fix box ticked is fixed at X in every spectrum.

## 3. Finding a parameter number

**Click and hold** a parameter's name in the parameters table: the label turns
into `p[12]` while the mouse button is down and returns to the name when you
release it. That is the number to use in `=[X,Y]`, in an `Expression` row and in
a `Distr`/`Corr`/`Recon` `par` field.

Numbering is flat and continuous over the whole table, starts at the baseline row
(`p[0]` … `p[7]`) and shifts whenever rows are added or removed — so read it off
again after editing the model.

## 4. Expression, Distr, Corr, Recon texts

These rows take **text**, not numbers:

- `Expression`: any formula in `p[i]`, e.g. `p[9]*0.5`, `sqrt(p[3]**2+p[5]**2)`,
  `33.0-p[12]`. The bare names `pi`, `e`, `exp`, `log`, `log10`, `sqrt`, `abs`,
  `power`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`,
  `tanh`, `arcsinh`, `arccosh`, `arctanh`, `floor`, `ceil`, `round`, `sign`,
  `mean`, `std`, `var` are available, here and in `Distr`/`Corr`; any other NumPy
  function is written `np.<name>`, e.g. `np.heaviside(X, 1)`. The result
  becomes a parameter of its own — click its label to read the number, then
  point real parameters at it with `=[X,Y]`. That is how a
  parameter is made to follow an arbitrary formula rather than a plain multiple.
- `Distr`: the probability-density text uses the variable `X` — the distribution
  axis running from `L` to `R` in `Num` steps, e.g. `X`, `exp(-X**2/2)`,
  `1/(1+X**2)`. Several `Distr` rows on the same model give a multidimensional
  distribution.
- `Corr`: the dependency text also uses `X` — the value of the distributed
  parameter — and sets a second parameter from it, e.g. `0.3*X` or `X**2`.
  That is how correlated distributions are built.
- `Recon`: same `par`/`L`/`R`/`Num` as `Distr`, but the shape is **fitted**
  instead of given. Leave the trailing `weights` field empty for a uniform
  start; the fit fills it in. `D_dif`/`D_dif2` (0…1) are the smoothness
  (Tikhonov) knobs on the first and second derivative of the reconstructed
  distribution — raise them if the result oscillates.

**Predefined expressions.** *Load expression* in a row's model menu opens the
expressions of the `Expressions` folder — each one a `.mdl` model file with a
single `Expression` row, its name the file name and its comment the `#@Comment`
lines, as in the Library. The chosen one's comment says what it does and what
each `{X}` of its text stands for; every `{X}`, and every `N1`, `N2`, …, gets a
box. What you type there (`p[12]`, a number) is put in; a box left empty keeps
the name, to be replaced in the table afterwards (an `N1` left as it is takes
each spectrum's own number in a sequence). **Load** inserts the result as an
`Expression` row at that row, the way the Library inserts a model; the `p[i]`
you type are the numbers the table shows before the insert, and are renumbered
with it. To keep an expression of your own, fill in name, comment and expression
and press **Save**. A formula in the comment written between dollar signs is
shown rendered.

Every one of these texts may also use the **parameters of the spectrum** (see
section 7.7): `N` is the spectrum's number in the sequence — its position in the
path box, counted from 1 — and `N1`, `N2`, … are the numbers given with it
(a temperature, an angle, …). So `p[9]+N*2` or `p[9]+0.01*N1`, linked into a
parameter with `=[X,1]`, forces that parameter to follow the spectrum number or
the temperature through a sequence. A single fit and **Show model** use the
first spectrum (`N` = 1); in a simultaneous one-model fit every spectrum uses
its own. A value a spectrum does not have blocks the start with a message, and
the names are refused in a model with `Nbaseline`.

The `par` field of `Distr`/`Corr`/`Recon` is **not** the flat `p[i]` number: it
counts the parameters of the model row it attaches to, `1` being that row's
second parameter (`δ` for most models). The amplitude (`T`, column 0) cannot be
distributed. Check your choice visually — the targeted parameter gets a grey
frame and becomes read-only, since the distribution now drives it, and the frame
moves as soon as you retype `par`.

`Distr`/`Recon` must sit directly under a fittable component, `Corr` under a
`Distr`/`Corr`/`Recon`; a wrong placement is refused with a message instead of
being accepted.

When several `Distr`/`Corr`/`Recon` rows follow the **same** component, each must
take a **different** `par`: they all write into the same parameter, so two rows
sharing a `par` would silently overwrite each other. **Show model** and **Fit**
refuse to start and redden the duplicate `par` fields (the first row keeps its
target — move the ones below it).

A parameter a `Distr`/`Corr`/`Recon` drives (the grey frame) holds no usable
number, and neither does the text column of one. Nothing may read them: a link
`=[i,Y]` to one and a `p[i]` in any `Expression`, `Distr` or `Corr` text are
refused when **Show model** or **Fit** starts. Put a shared value (a centre, a
width) in a `Variables` row; to make another parameter follow the distribution,
use a `Corr`.

The Expressions are calculated one by one from the top of the table, so an
Expression may use any parameter but only the Expressions above it; one that
uses an Expression below it is refused when **Show model** or **Fit** starts.
A link may point at another link — the chain is followed to the parameter that
holds a number — but a chain that goes round in a loop is refused. In an
Expression, use the source of a link: if parameter `Z` is linked as `=[X,Y]`,
write `p[X]`, not `p[Z]`.

An unparsable text, or an empty parameter field (typically left behind when a
value referenced by `=[X,Y]` was deleted), blocks **Show model** and **Fit** with
an explicit message and reddened fields rather than failing halfway through.

## 5. Keyboard and mouse

| Action | Shortcut |
| --- | --- |
| Fit | `F5` |
| Show model | `Enter` |
| Take result as new model | `F8` |
| Zoom the plot at the cursor | mouse wheel over the figure |
| Apply typed exclusion regions | `Enter` in their box (not Show model) |
| Cancel *Pick on plot* | `Esc` |
| Clean model (baseline kept) | double-click the *Clean model* button |
| Replot a stored fit | click its row button in the results table |
| Find a word in a help window | `Ctrl+F`, then `Enter` / `Shift+Enter` |

## 6. Parameters table details

- The **fix** check box next to each name freezes that parameter. Orientation
  angles, texture parameters and the like start fixed on purpose — untick them
  only when the data can support them.
- A check box that is **greyed out with a lock icon** is structural and can never
  be fitted: `par`/`Num` of `Distr`, `par` of `Corr`, `par`/`Num`/`D_dif`/`D_dif2`
  of `Recon`, `S` of `Relax_MS`, `Num` of `ASM`, `N/Γ` of `SCDW`.
- The two small fields under each value are the **lower and upper bound**. Leave
  a bound empty for "unbounded". A value outside its own bounds (a value on a
  bound is inside) blocks **Fit** with a message and a reddened field.
- **Take result as model** writes fitted values with at most four decimals and
  never in `1e-05` notation; a value the fit did not move (fixed, a preset, or
  stopped on a bound) keeps all its digits. The bounds come back as the fit had
  them.
- Switching **CMS** on turns a baseline `Nnr` of `0` into `=[0,0.67]` (the
  non-resonant counts as 0.67 of `Ns`); switching back to **SMS** turns exactly
  that link into `0` again. Any other `Nnr` is left as you set it.
- In CMS and SMS alike, `Onr`, `c²nr`, `linnr` of the baseline start linked to
  its `Os`, `c²s`, `lins` (`=[1,1]`, `=[2,1]`, `=[3,1]`), and those of every
  `Nbaseline` row to its own. These links are the correct physical approach for
  quanta from the source. However, if the detector has electronic (static)
  noise, `c²nr` should be variable and smaller than `c²s` — which can be
  achieved with `Expression` and `Variables` rows: e.g. a `Variables` value
  between 0 and 1 for the source's part of `Nnr`, an `Expression` `p[2]*p[k]`
  (`k` the number of that value) and `=[X,1]` in `c²nr` (`X` the number of
  the `Expression`).
- Both baseline polynomials are centred: `lins` and `c²s` multiply `(v − Os)`
  and `(v − Os)²`, `linnr` and `c²nr` multiply `(v − Onr)` and `(v − Onr)²`.
- Empty rows between models are harmless — they are dropped when the model is
  saved.
- Selecting `Nbaseline` turns its row light grey (silver): it opens the next
  spectrum's section, it is not a component.
- `Layer` is a marker, not a component: components above and below it are
  propagated as separate layers of the sample (correct for a stack of different
  orientation or of different phases, where the transmission matrices do not
  commute).
- `Library` in the model dropdown opens the phase library and `Load model` a
  `.mdl` file browser; both ADD the chosen model's components to the current one
  at that row (your baseline is kept, links to it are preserved, links between
  the added parameters are re-indexed). `Copy`/`Paste`, `Insert`/`Delete` act on
  the row you picked them from.
- The light/dark theme toggle lives in the *Supp* menu (first entry).
- *Supp* → **Set parameters of Be (optics impurity)** / **of KB (Nanoscope
  impurity)** edit the two impurity Doublets (`Be.txt`, `KB.txt`): the current
  state of the beamline, used by the `Be`/`KB_nano` model entries, the
  calibration and the results table, which lists a row equal to one of them as
  *Impurity* and leaves it out of the percentages. Rows already in the table keep
  their numbers — pick the entry again to load new ones; for old data, set the
  values the beamline had then.

## 7. Spectrum files and instrumental function

### 7.1 The instrumental function in .dat files

- A `.dat` file may carry the instrumental function in its header — every line
  above the first data line (one that starts with a number):
  `#@GCMS G Δθ d D` marks a CMS spectrum (its source: Gaussian width, cosine
  smearing, the source and aperture diameters d and D in mm; a file written
  before carries G alone: no smearing), `#@INSexp` (sum of Gaussians) or
  `#@INSth` (theoretical shape), with `#@INSint`, an SMS one.
  SYNCmoss writes those lines when it converts a spectrum — for SMS only the
  instrumental function **in use** — and reads them back per spectrum, so a
  batch may mix CMS and SMS files, and Gaussian and theoretical ones. Switch the
  behaviour with *Instrumental function -> Switch to the global instrumental
  function* (and back with *Switch to the instrumental function from .dat
  files*); the button's caption says which is in use.
- When the header of a spectrum selects the other mode than the CMS/SMS box (a
  `#@GCMS` file fitted in SMS mode, or the reverse), the header wins: Fit names
  those spectra and asks before it starts. To fit them in the mode of the box,
  switch to the global instrumental function, or remove the lines from the files.
- *Instrumental function -> Overwrite instrumental function in selected .dat
  files* writes the instrumental function in memory — for the CMS/SMS mode
  selected, the lines RAW → .dat conversion writes — into every `.dat` file of
  the path box: it replaces the `#@` lines a file has, or adds them just above
  its data. It asks first, and nothing else in the files changes. Other files
  (`.mca`, …) are left alone, and the calibration file (`Calibration.dat` of the
  parameters folder, or another one chosen) is never written.

### 7.2 The CMS source: G, Δθ, d and D

- **G** and **Δθ** (under the CMS/SMS boxes) describe the CMS source; SMS
  spectra ignore them. **G** is the Gaussian width (FWHM, mm/s) of the source
  line. **Δθ** (rad, 0 = off) is the *cosine smearing*: a photon leaving the
  source at the angle θ to the drive axis sees the drive velocity times cos θ.
  The velocity axis is the **transducer velocity**, so with Δθ a line moves out
  and broadens in proportion to its distance from the velocity at which the
  drive is at rest — the middle of the spectrum's velocity range. At Δθ = 0.2
  an outer α-Fe line sits 1 % farther out (⟨1/cos θ⟩), its area is 1 % larger,
  and it gets 0.03 mm/s rms wider. A laser calibration gives the transducer
  velocity; an α-Fe calibration gives it when made with the same Δθ (it uses
  the Δθ field) — calibrated at Δθ = 0 and fitted at Δθ = 0.2, every velocity
  is 1 % off. Δθ is the angle of the point source whose cos θ has the same
  spread — atan(R/L) at the distance L from an aperture of radius R,
  atan((R⁴ + 6R²r² + r⁴)^¼ / L) for a source of radius r; the disc itself is
  computed exactly, at any angle. The integration keeps up with it to
  Δθ ≈ 0.5 rad at JN 64 (≈ 0.3 at JN 32); beyond, the cyan line wiggles:
  raise JN. Both come from the defaults at start, are
  written into `#@GCMS` of converted `.dat` files, and a file's own values are
  used for it. The fields show 4 decimals and keep every digit (the tooltip
  shows them all).
- *Supp* → **Set CMS source and aperture diameters**: d, the source's active
  spot, and D, the aperture — whatever limits the rays reaching the detector,
  usually the sample or a lead mask on it. Only their ratio d/D matters: it
  sets the shape of the smearing, Δθ its width, and it has a small effect —
  with Δθ fitted, a thick α-Fe spectrum changes by 0.2 % of its dip from a
  tiny source to d = D, by < 0.1 % with d/D off by a factor of 2. d/D and D/d
  act alike: a large source behind a small aperture (d/D = 10) is nearly a
  point, as a small one behind a large aperture is. Both must be above 0 (no
  source is a point). The default, 5 and 20 mm (d/D = 0.25), is the least bad guess when
  the geometry is not known (≤ 0.11 % of the dip whatever it is). Saved with
  the defaults at once, and written into `#@GCMS` with G and Δθ.

### 7.3 The Instrumental function button

- The **Instrumental function** button holds every instrumental-function
  action: *Refine* (single line, pure α-Fe, model) first, then *Find*, then the
  Δθ searches, the `.dat` file options, *Save G, Δθ, d and D as defaults* (the
  start values of the next session) and *Reset instrumental function to
  default values* — of the one in use: in CMS G = 0.09 mm/s, Δθ = 0.1 rad,
  d = 5 mm, D = 20 mm (saved and shown); in SMS the selected description. Its
  caption says where a spectrum's instrumental function comes from: *(use IF
  from file)* — the spectrum's own `.dat` header when it has one — or *(use
  global IF)*. Every *Find* and *Refine* writes its result at the end: CMS to
  the defaults (G, Δθ, d, D), SMS to its description's files.
- *Refine* is usually the better choice: it continues from the shape already
  fitted to this source — faster, and it cannot wander as far. *Find* discards
  the stored instrumental function and restarts from the built-in values.
- In CMS there is only *Refine*: it fits G from its field, Δθ held. With a
  single line the absorber is a Singlet at the natural width, its T and shift
  free; Ns and Nnr are free. *Refine Δθ* (CMS only) fits Δθ from its field
  (0.15 rad when it is 0) with G and d/D held. *pure α-Fe* first asks for the
  source — G, its f_LM, C_high (the 122/136 keV quanta counted per point) and
  the static noise of the detector — and sets the non-resonant baseline to
  Nnr = Ns (1 − f_LM)/f_LM + C_high, curving with Ns (linked), with the static
  noise added flat; T, the texture A, Ns and its curvature are free, and so is
  the field, within 1 T of the α-Fe one: the calibration has already used the
  line positions, so Δθ comes from the line shape. A field off 33.04 T means
  the axis is off by as much — calibrate again with the Δθ found. Find G and f_LM first
  from a single-line spectrum, where Δθ hardly matters, and C_high as the count
  rate with a 100 µm Cu foil in the beam times the counting time of one point,
  less the static noise. *model* fits the table's model with its links as set.
  Every CMS search stores G and Δθ as the defaults.
- *Calibrate + find G, Δθ: α-Fe + single line* (CMS) does the calibration and
  the source in **one** fit: put the raw α-Fe and the raw single line, measured
  with one source, geometry and drive, first and second in the path box. A
  dialog takes where G, Δθ and f_LM start (G and Δθ from their fields) and, for
  each spectrum, its C_high and static noise per channel — they differ when the
  counting times do. The calibration with the start values gives the drive to
  start from, and the single line its thickness; then the drive, its count
  distortion, G, Δθ, f_LM and both absorbers (α-Fe: T and texture at δ 0 and
  the α-Fe field; the single line: T and shift; both at the natural width, no
  Gaussian of their own) are fitted together. It writes `Calibration.dat`,
  sets the G and Δθ fields and stores them as the defaults; the plot shows
  both spectra folded with their fit.
- *Find: model* / *Refine: model* fits the model of the table together with
  the instrumental function: its free parameters are fitted too, and links
  `=[X,Y]` and Expressions follow their sources. Afterwards the results table
  shows the model as after a fit — *Take result* takes it back, *Save result*
  does not save it (fit to save a result). With several spectra in the path box
  they are all fitted with **one** instrumental function — there is no
  sequence of searches: a model with `Nbaseline` rows as built (one spectrum
  more than rows), any other model as the simultaneous one-model fit expands it
  (an `=(X)` gets a value per spectrum, every spectrum its own baseline). Each
  spectrum gets its own panel of the plot. A model with a `Recon` is refused.
  The single-line and α-Fe searches are single-spectrum procedures: they use
  the first spectrum. Every search leaves out the exclusion regions (section 8).

### 7.4 SMS: theoretical or Gaussian instrumental function

- *Supp -> Choose how to approximate instrumental function* chooses what the SMS
  *Find* and *Refine* fit:
  - **Theory**: the simulation of the ⁵⁷FeBO₃ source — a few PHYSICAL numbers
    instead of empirical ones, and the proper tails. Technically the proper
    way, but slower calculations.
  - **Set of Gaussians** (default): the empirical description, a free sum of
    Gaussians (their number is set beside the button). Faster and always able
    to follow the data, but the parameters mean nothing physical and a
    Gaussian sum has short tails. So far no issue was found by using it.

  Spectra that carry their own instrumental function in the `.dat` file are not
  affected by this setting.
- With **Theory**, *Find* and *Refine* fit the *theoretical* SMS instrumental
  function instead of a free sum of Gaussians: the simulated energy distribution
  of a ⁵⁷FeBO₃ synchrotron Mössbauer source, with four physical numbers —
  the rocking-curve position θ [µrad], the staggered hyperfine field B_s [T]
  (the temperature knob), the quadrupole splitting ΔE_Q [mm/s] and the source
  shift [mm/s]. Same three reference absorbers as the ordinary search. It is
  slower (≈100 s against ≈15 s) and, having 4 free shape numbers instead of 9,
  will usually give a slightly *higher* χ² — what it gives back is parameters
  that mean something and the correct v⁻⁴ line wings, which a Gaussian sum
  cannot have at all.
- The theoretical *Find* first asks for its starting values — change them if
  this source was run far from the usual settings:
  - θ, the position you set on the rocking curve: held during the search —
    released from one spectrum it is invented, not measured;
  - ΔE_Q, a property of FeBO₃: held; −0.3900 comes from the 83-spectrum fit;
  - f_LM, f_LM × enrichment of the source crystal: held; 0.7718 is both the
    Debye value and where the 83-spectrum fit puts it.

  B_s and the velocity shift are not asked: Find maps both of them.
- The theoretical result is stored in `parameters/INSth.txt`, next to the sum of
  Gaussians in `INSexp.txt`; *Supp -> Choose how to approximate instrumental
  function* selects the one in use, and switching costs nothing. A spectrum
  converted from RAW carries only the one in use then — `#@INSth` or
  `#@INSexp` — and its fits use that one whatever the setting is later, so
  spectra converted with either can be fitted together (sequence or
  simultaneous), each with its own. A `.dat` written before this carries both;
  for it the setting chooses. The ordinary (Gaussian) *Refine* of a spectrum
  that carries only `#@INSth` starts from `INSexp.txt`.
- *Supp -> Plot instrumental function from memory / from spectrum* draws what is
  actually in use — from `INSth.txt`/`INSexp.txt`, or from the loaded
  spectrum's own header — with its FWHM, centre and first moment, on a linear
  and a logarithmic scale. The log panel is where a theoretical and an empirical
  instrumental function stop looking alike. It opens in its **own window**, with
  its own zoom toolbar and a Save button, and stays there while you work on the
  spectrum; the main plot is never touched.

### 7.5 The calibration file

- The calibration file's first line is `# <method> <n1> <n2>`: `sin` or `lin`
  folding and the raw-channel range. Without it, `sin` folding over all channels
  is assumed.

### 7.6 Several spectra

- Several files in the path box (a Python-style list, or just comma-separated)
  are fitted the way *Multispectra settings* says (the mode is shown on the
  button): a **sequence** — each spectrum on its own, starting from the initial
  guess or from the previous spectrum's result — or a **simultaneous one-model
  fit**. A model with independent `=(X)` values is always fitted the second way.
  A model with `Nbaseline` rows is fitted **simultaneously** as you built it
  (one `Nbaseline` per extra spectrum), with a sequence mode chosen. Without
  `Nbaseline` rows, **Show model** draws the model on the first and the last
  spectrum, each with its own baseline and `N`, `N1`, … — in every mode.
- *Multispectra settings → Fit one spectrum…* fits ONE spectrum of the path box
  on its own, with the model of the table and that spectrum's own `N`, `N1`, …:
  give its number (its place in the path box) or its name (the file name without
  `.dat`) — each field follows the other. A name that is not in the path box is
  looked for next to its spectra (such a file has no `N`, `N1`, … of its own).
  Nothing is saved: **Save result** does that. Not for a model with `Nbaseline`
  rows; with fewer than two spectra it only says so.
- The **simultaneous one-model fit** fits the model of the table — built for ONE
  spectrum — to all the spectra at once. Every free parameter has one value
  shared by all the spectra, except the independent `=(X)` ones (section 2),
  which get a value per spectrum. Every spectrum has its own baseline: its `Ns`
  starts from its own counts (as in a sequence), the other values are the
  table's, and a link in the baseline (e.g. `Nnr = =[0,0.67]`) stays inside each
  spectrum. A link to an independent or fixed parameter follows that spectrum's
  own copy, a link to a shared one the common value. `N`, `N1`, … are every
  spectrum's own. The model must not have `Nbaseline` rows (the fit makes every
  spectrum's section itself). There is one χ² for all the spectra.
  - The main window shows the **first and the last** spectrum; a separate
    **result window**, opened at the end of the fit, shows every one of them,
    chosen with the slider at its bottom, with that spectrum's `N`, `N1`, … and
    its part of the results (no correlation matrix: one spectrum's part of a
    joint fit has none of its own).
  - The results table lists the whole fit: every spectrum's baseline and
    components, as if you had built the model with `Nbaseline` rows. A click on
    a component of a spectrum the main window does not show offers to open the
    result window there.
  - *Take result as model* puts back only the model for one spectrum: the first
    spectrum's values, and every link and independent value as it was when the
    fit started — the X of an `=(X)` stays the start value it was.
  - `Recon` is not recommended in such a fit: its weights are always shared, so
    every spectrum gets the same reconstructed distribution — if the
    distribution really changes from spectrum to spectrum (with temperature,
    say), expect a compromise shape that fits none of them well, and a slower
    fit.
- The **result window** also opens at the end of a **sequence** — every fitted
  spectrum with its own χ² and its own correlation matrix — and of an
  **`Nbaseline` fit** — every spectrum with its own components, the fit's one χ²
  and no correlation matrix.
- In the result window a formula (`Expression`, `Distr`, `Corr`) reads as in the
  whole fitted model — after a one-model fit with that spectrum's own `p[i]` and
  its `N1`, … filled in — and is evaluated with the whole fit, so one spectrum's
  formula may use another spectrum's parameters.
- Before a sequence starts, Fit asks for confirmation and shows the save path:
  a sequence saves as it goes (section 10).
- When the spectra do not share one instrumental function (CMS and SMS spectra
  mixed, or `.dat` headers that differ), Fit says so and asks before it starts;
  *Do not ask again* silences the question for the rest of the session.

### 7.7 Parameters of the spectra

- A spectrum in the path box may carry parameters of its own, used as `N1`,
  `N2`, … in the formulas (section 4): write it as a tuple,
  `[('Fe_4K.dat', 4.2, 0), ('Fe_77K.dat', 77, 0)]`. Plain paths and tuples may
  be mixed.
- *Multispectra settings → Load parameters of the spectra* fills those tuples in from
  a text file, one parameter per line, one number per spectrum, separated by
  spaces and/or tabs. Lines starting with `#` are comments (e.g. `#N1`), except
  two reserved names: `#basename`, whose next line names the spectrum of each
  column (the numbers then go by name, not by position), and `#number`, whose
  next line gives each spectrum's `N` and reorders the path box accordingly —
  it needs `#basename`. Without `#basename` every line needs at least one number
  per spectrum. When the file does not fit, nothing changes and the log says
  why. *Save parameters of the spectra* writes the path box as it is now, always
  with `#basename` and `#number`:

      #basename
      Fe_4K.dat	Fe_77K.dat
      #number
      1	2
      #N1
      4.2	77

### 7.8 Other spectrum sources

- A path ending in `/` or `\` is read as a folder: the first `.dat`/`.mca` in it
  is used.
- On Linux with the `bliss` package installed (it is not a dependency), a Bliss
  channel name `McaAcq_channel_<...>` can stand in for a file: the spectrum the
  MCA is accumulating is read from Bliss and folded like an `.mca` file with the
  current calibration, afresh on every **Show spectrum**, **Show model** and
  **Fit**. The beacon server is taken from the `BEACON_HOST` environment
  variable, or is `id14:25000` when that is unset.

### 7.9 Channel average

- While a point is counted the drive runs through a range of velocities — the
  point's **channel**: the velocity step of a triangular drive; for a sine, a
  width shrinking to nothing at the turning points. With *Supp → Average model
  over each channel* ON (the default) every point is the model averaged over its
  channel; OFF, the model at the point's velocity, as before. Fits, Show model,
  the instrumental-function searches, the calibrations and *Create spectrum from
  model* all follow it; *Save settings as default* keeps it. Switching it warns:
  an instrumental function or a calibration found with the other setting does
  not fit — redo them, or *Switch back*.
- Averaging the model over the channel is the same as averaging the source line
  over it (for any absorber), so the source is averaged before the transmission
  integral, as for the cosine smearing, and a calculation takes about as long as
  without. With Δθ, each ray sees its channel times cos θ.
- The widths come from the drive: every point is counted for the same time, so
  a spectrum's velocities, sorted, are a line (triangular drive: every channel
  one step wide) or a cosine (sine drive: the channel at the velocity v is
  w √(A² − (v − c)²) wide, 0 at the turning points), fitted once to the whole
  axis. A point's width depends on its velocity alone — exclusion regions and
  the ends need no care. An axis with points missing (saved by another
  program) is neither: Fit, Show model and the searches are not started, and a
  window says what to do — calibrate and save the spectrum in SYNCmoss (it
  keeps the velocities of excluded points, section 8), or turn the average off.
- The effect is largest for narrow lines and wide steps (wide velocity ranges):
  at a step of 0.08 mm/s a thin line at the natural width comes out a few % of
  its depth too deep without the average. An instrumental function found with
  the average OFF already contains the channel — it is broader by about 0.68 ×
  the step, in quadrature — and with it ON the channel would count twice: find
  the instrumental function (G for CMS, the Gaussians or the theoretical shape
  for SMS) again with the setting you fit with.

## 8. Exclusion regions

Velocity regions whose points are **not fitted**: they are left out of χ², the
degrees of freedom and the errors. Switch them on with **Apply exclusion** in
the plot toolbar; a bar appears above the plot.

- Type the regions in mm/s as `lo:hi; lo:hi`, e.g. `-3:-2; 1:2` — only `;`
  separates regions, and the two ends may come in either order. **Enter** (or
  leaving the box) applies them: they are sorted, overlapping ones merged, and
  the box shows the result. A point exactly on an end is excluded.
- **Pick on plot**, then click the two ends of a region on the plot (either
  order); **Esc** cancels. It works on whatever is on screen — spectrum, model
  or fit — but not while the toolbar's Pan or Zoom is on. A clicked end is put
  between the two data points around the click, with no more decimals than
  needed to tell them apart.
- The regions are grey bands on every spectrum plot: the main window, the
  result window and the saved pictures.
- **Apply exclusion** off: the regions are not shown, not used and not saved
  (the text stays for the next time). It is off at every start.
- The same regions apply to every spectrum (sequence, `Nbaseline`, one-model).
  For regions that only some spectra need — or that should stay for good — use
  *Change spectrum(a) → Save spectrum without excluded points*: every selected
  spectrum is written as `excl_<name>.dat` with the counts of those points
  `nan` (one spectrum where you say, several into the save path's folder): no
  fit takes them, and the velocity axis stays whole — where the drive is at
  rest (the cosine smearing's centre) and the drive itself (the channel
  average's widths) stay those of the measurement. A `.dat` keeps the
  instrumental function of its header; an `.mca` (or other raw file) is
  calibrated as it is read, so its `.dat` gets the instrumental function in use
  — CMS or SMS, as the check boxes say — as *RAW → .dat* conversion writes it.
- Changing the regions (typing, picking, *Load*, or **Apply exclusion**)
  changes the bands of the plot on screen. If it shows a fit, that fit is no
  longer a result of these regions: its χ² disappears, the results table is
  emptied, and it can be neither saved nor taken as model (**F8**) — fit again.
- A fit is not started when no more points are left after the regions than
  there are free parameters; a sequence checks all its spectra before it starts.
- **Clean** removes every region (**Apply exclusion** stays on).
- **Save** / **Load** write and read the regions as a small text file
  (`<name>_exclusion.txt`); **Load** replaces the current regions.
- The background (`Ns`) a sequence starts from is still estimated from the
  whole spectrum.

## 9. Errors, χ² and the correlation matrix

- Every fit of counts — a spectrum, the calibration, the instrumental-function
  searches — maximises the **Poisson likelihood**: every channel counts with the
  inverse of the counts the model expects there. For counts that is the most
  precise fit there is, and an unbiased one.
- The result shows two numbers, both divided by ν = points − free parameters:
  **D**, the Poisson deviance (what the fit minimises), and **χ²**, Pearson's
  Σ (y − m)²/m. With a right model and Poisson counts both are about 1, within
  ±√(2/ν), written in brackets in units of the last decimal:
  `D = 0.884(63), χ² = 0.885(63)` means ±0.063 (500 points). D and χ² differ only
  when a channel has few counts.
- An error is the statistical 1σ of the fit times √χ², so it follows the noise
  the spectrum really has: a χ² of 0.88 makes the errors 6 % smaller, one of 2
  makes them 41 % larger. **Supp → Empirical correction of errors by χ²**
  turns the factor off (ON by default): the errors are then those of exactly
  Poisson counts.
- The % intensities and the Expression rows get their errors from the same
  covariance, with the same factor; the correlation matrix does not depend on it.
- A parameter the other free ones reproduce exactly — two that enter only as a
  product, an offset added to the baseline — gets no error, as if fixed, and the
  log names it. *Take result* fixes it; the earlier one keeps their combination
  free.
- The errors are statistical only: the calibration, the instrumental function
  and the choice of model are not in them.

## 10. Saving

**Save result** writes several files next to the save path, not one. `<base>` is the
save path without the spectrum's extension: `Fe_4.2K.dat` gives
`Fe_4.2K_param.txt`, and a dot inside the name itself (the `4.2K`) is kept.

- `<base>_param.txt` — parameters, errors, χ²; when the fit used the parameters
  of the spectrum, `N`, `N1`, `N2`, … follow the file name
- `<base>_inputs.txt` — those parameters as the fit used them, in the
  format of *Load parameters of the spectra* (only when there are any; a
  sequence writes one for its whole run when it starts)
- `<base>_graf.txt` — the plotted curves: velocity, data, baseline, fit, then
  one column per component (and the distribution curves, if any). A simultaneous
  (`Nbaseline`) fit writes that whole block for every spectrum, prefixed
  `S1_`, `S2_`, … — each with its own baseline — padded with `nan` to the
  longest spectrum
- `<base>_combo.png` — the figure together with the rendered results table
- `<base>.svg` — the figure alone
- `<base>_distributions.png` — every `Distr`/`Corr`/`Recon` curve, when present
- `<base>_result_model.mdl` — the fitted model: the model the fit started from,
  with every free parameter at its fitted value (the numbers of `_param.txt`)
  and the links, fixed values, bounds and expressions exactly as they were
  fitted; a `Recon` gets its fitted weights. Changing the table after the fit
  does not reach it. Loading it puts the result back into the table, ready for
  another fit. Its own name, so it never overwrites the `.mdl` you are working
  on with **Save model**. It is saved last and asks its **own** overwrite
  question, separate from the one for the result files: answering "no" there
  keeps the older file and leaves the five result files as they were just
  written. (A sequence writes one for every spectrum, under the spectrum's own
  name, without asking.)
- `<base>_exclusion.txt` — the exclusion regions the fit left out (section 8),
  ready for *Load*; only when they were applied. Then `_param.txt` also has a
  last column `Exclusion regions` with the same text, and `_graf.txt` a last
  column `Fitted` (`S1_Fitted`, … for an `Nbaseline` fit): 1 for a fitted point,
  0 for an excluded one. A sequence writes `<base>_exclusion.txt` once, when it
  starts.

When `<base>_param.txt` already exists, **Save result** asks: *Save* appends to
`_param.txt` (and to the `_result_table_PNG.html` of a one-model fit) and
overwrites the other files, *Discard* overwrites them all, *Cancel* does nothing.

A **sequence** saves as it goes, without asking: ONE `<base>_param.txt` — one
line per fitted spectrum — and ONE `<base>_result_table_PNG.html` with the
pictures of every spectrum (figure + table, and the distributions when there are
any). Both are **extended** by every run: each run adds a heading with its date
and time to the page, and a new line of names to the parameter file where the
columns change. So a large batch can be fitted group by group, each group from its own
initial guess, into the same two files. The curves, figure, pictures and fitted
model of each spectrum are also saved next to them under the spectrum's own name
(`Fe_4K_graf.txt`, `Fe_4K_combo.png`, `Fe_4K_result_model.mdl`, …).

A **simultaneous one-model fit** saves the same way, but only when you press
**Save result**: ONE `<base>_param.txt` with a line per spectrum (the model's
parameters for that spectrum — the shared ones repeat on every line — and the
fit's one χ² on every line), ONE `<base>_result_table_PNG.html` (both extended
when you answer *Save* to the "file exists" question), the `<base>_inputs.txt`
of all the spectra, and every spectrum's curves and pictures under its own name.
`<base>_result_model.mdl` is the model for one spectrum, the independent `=(X)`
values with the X they started from — ready for another fit.

## 11. Other

- **! INTERRUPT !** does not just stop the fit: it terminates the worker pool and
  builds a fresh one. Use it if a calculation is stuck — the fit in progress dies
  with it, but the model and the loaded spectrum are untouched.
- *Supp -> Set SMS polarization* sets the SMS beam linear-polarization degree
  (0 = unpolarized, 1 = fully polarized; default 0.98). It changes the readout of
  every thick model.
- *Supp -> Set number of points for full transmission integral* is the
  accuracy↔speed knob of the thickness integral (32 for SMS, 64 for CMS;
  switching the CMS/SMS box moves it between those two defaults for you).
  Whether it is high enough is shown, not guessed: the **cyan
  "Integration check (×4)"** trace under every Show model and Fit is the same
  model recomputed with four times as many points, minus the one you see. A flat
  line means converged; a visible wiggle means raise the number.
- *Supp -> Save settings as default* writes the current settings to
  `config.json` in the parameters folder, and every later start of SYNCmoss uses
  them: CMS or SMS, the velocity direction, the theme, where a spectrum's
  instrumental function comes from (its `.dat` file or the global one), how the
  SMS one is approximated (theory or Gaussians, and how many), the χ² correction
  of the errors, the integral points for SMS and CMS, the polarization and the
  Multispectra mode. Not among them: G, Δθ, d and D, which have their own
  defaults (*Instrumental function -> Save G, Δθ, d and D as defaults*), and the
  exclusion regions, off at every start. *Supp -> Restore default settings*
  deletes the file: the open window keeps its settings, the next start has the
  built-in ones.
- *Change spectrum(a)* holds five destructive-looking but useful operations:
  sum all listed spectra, subtract the current model from the spectrum, create
  a spectrum from the model, halve the number of points, and save the spectra
  without the points of the exclusion regions (section 8).
- *Create spectrum from model* calculates the model of the table, without
  noise, on the velocities **Show model** uses: those of the first spectrum in
  the path box (its own column for a `.dat`, `Calibration.dat` for an `.mca` or
  other raw counts), or the `Model_<range>` grid. It runs in the background
  (**! INTERRUPT !** stops it), then asks where to save the `.dat`. Its header
  carries the instrumental function the model is calculated with — the
  spectrum's own `.dat` header or the one in memory, as the settings say. The
  file is not loaded. Not for a model with `Nbaseline` rows; exclusion regions
  are ignored.
- Distributions are drawn in the plot area: the **Distribution / Spectrum**
  button toggles the view once a model with `Distr`/`Recon` has been shown.
- The help windows and the License window have a **Find** box: **Ctrl+F**,
  type a word, **Enter** / **Shift+Enter** for the next / previous match, **Esc**
  empties the box. The search ignores case and starts at the text cursor — the
  top, or where you last clicked in the text.

## 12. When something goes wrong

Any error that reaches the terminal — whether SYNCmoss caught it or not — also
opens a **report a bug** window with the message in it and a *Copy the error
message* button. The terminal output is unchanged; the window is just a copy,
and it matters mostly for the packaged builds, which have no terminal at all.

- The program usually keeps working afterwards. Tick *do not show this window
  again* to silence it until the next restart.
- Please do send the report: the message alone is rarely enough, so add what you
  were doing (which button or action) and attach the spectrum and the `.mdl`.
- *Supp -> **Contact the author*** (last entry) shows the address and the issue
  tracker — for bug reports, feature requests and questions alike. Nothing here
  opens a mail client for you: copy the address and write from wherever you
  normally do.

## 13. Models: details

The longer notes on the models. Every parameter of every model, with a short
definition, is in *Supp -> Models description*.

### 13.1 Doublet and Sextet

- Sextet GH: the Gaussian spread of H, in T. It is added in quadrature to G with
  each line's own field sensitivity, so the outer lines broaden most — a field
  distribution, not an extra uniform width.
- Sextet a+: extra symmetric splitting of the OUTER pair only (line 1 by $-a_+$,
  line 6 by $+a_+$); the second-order/relativistic correction of lines 1 and 6.
  a-: the same for the four inner lines (2 and 4 by $+a_-$, 3 and 5 by $-a_-$).
- Sextet I1/I3: the intensity ratio of the outer (1, 6) to the inner (3, 4)
  lines; 3 is the isotropic 3:2:1 case. The middle pair (2, 5) keeps its
  isotropic weight and the total area is conserved.
- Doublet G2/G1: the Gaussian width of line 2 as a multiple of G — for an
  asymmetric doublet. 1 = both lines equally broad.

### 13.2 Relax_MS

Many-state superparamagnetic relaxation: the moment of one particle of spin S
hops between its $2S+1$ projections in a uniaxial anisotropy potential, and the
$\Delta m = \pm 1, 0$ line groups are broadened by the resulting stochastic
process.

- H is the saturation hyperfine field, i.e. the field of the fully aligned
  ($m = S$) state. State $m$ sees $H\,m/S$.
- The model has NO Gaussian width parameter — the broadening it produces is the
  relaxation itself, not a convolution.
- R, the relaxation rate, is the prefactor of the jump rates
  $\propto R\,[S(S+1)-m(m\mp1)]$. Small R = a static sextet, large R = a
  collapsed (fast-relaxing) line.
- alfa, the reduced uniaxial anisotropy barrier (the barrier energy in units of
  $k_BT$), enters as the Boltzmann factor of the UPWARD jumps only — jumps
  toward lower energy keep a factor 1, which is detailed balance. alfa = 0 is
  free hopping; large alfa freezes the moment near $m = \pm S$.
- S, the spin of the particle — the total spin quantum number, giving $2S+1$
  projections — is the physical size knob of the superparamagnetic particle
  (and therefore also what sets the cost: the model solves a $(2S+1)$-state
  system per velocity point). Structural, so it is never fitted.

### 13.3 Relax_2S

Two-state (Blume) relaxation: the nucleus jumps between two complete hyperfine
sets. Set H2 = -H1 for a moment that reverses; set the two sets differently for
a jump between two chemical/magnetic states.

- Am: a mirror-symmetric pair (H2 = -H1 at P1/P2 = 1) is unmagnetised: the
  Faraday term then cancels for any Am, which is correct physics, not a bug.
- $\Omega_{12}$, the transition rate between the states, is in the same (mm/s)
  units as L, so $\Omega_{12} \ll L$ is slow relaxation and $\Omega_{12} \gg$
  the splitting is the fast (collapsed) limit.

### 13.4 Hamiltonian

Full Hamiltonian model (magnetic + quadrupole, arbitrary orientation) of a
MOSAIC textured crystal. It replaces the former single-crystal `Hamilton_mc`
(now $A = Am = Ah = 1$) and powder `Hamilton_pc` (now $A = Am = Ah = 0$),
both of which it reproduces exactly, and interpolates continuously between them.
`Hamilton_mc` and `Hamilton_pc` are no longer offered in the model dropdown;
opening an old model file rewrites them to `Hamiltonian` with these order
parameters. They will be removed in a later release.

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

**Reading the geometry in CMS and in SMS.** The **same** 15 parameters are used
in both source modes; what changes is which lab axis $(\theta, \varphi)$ points
at, and that changes what is measurable.

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

### 13.5 ASM: the angle ω

$\omega$ is the cycloid-plane angle about the easy axis. $(\theta_k, \varphi_h)$
fix the easy (anharmonicity) axis $\mathbf{u}$, which always lies IN the plane
the moment rotates in. $\omega$ picks the second in-plane axis
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

Num, the number of sampling points over one modulation period, is rounded
internally to $6k+1$.

### 13.6 SCDW

Spin/charge density wave. The spin AXIS is fixed at $(\theta_k, \varphi_h)$;
only the scalar hyperfine parameters (signed field magnitude, central shift,
quadrupole shift) are modulated along the wave, sampled at a fixed number of
phase points over one period and averaged as a thick (polarized) sextet. The
field keeps its SIGN, so a reversed moment automatically swaps the line positions
within the (1,6) and (3,4) pairs. Similar to the SpectrRelax SDW/CDW parameter list.

N/Γ, the grid resolution, is the number of grid steps per line width used to bin
the wave positions. This is the accuracy↔speed knob (larger = finer grid = more
Voigt evaluations = more accurate = slower; error ~ 1/(N/Γ)). Default 4.

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

### 13.7 Layer, Library and the other row actions

- `Layer`: components above and below it are exponentiated as separate layer
  matrices, multiplied in order at the amplitude level. Use it for a genuine
  stack (two foils, a coating on a substrate, differently oriented crystals) —
  components within one layer are averaged before the exponential instead. For
  scalar-only models it behaves like a no-op.
- `Library`, `Load model`, `Copy`, `Paste`, `Insert`, `Delete` are not models:
  they act on the table row they were selected from. `Library` opens the phase
  library and `Load model` a `.mdl` file browser; both ADD every component of
  the chosen model at that row, keeping your current baseline and re-indexing
  the added `p[i]` / `=[X,Y]` references (links to the baseline's first 8
  parameters stay as they are). `Copy`/`Paste` move a row (model, values and
  fix states) through an internal clipboard; `Insert` and `Delete` add/remove a
  row and renumber every `p[i]` and `=[X,Y]` reference for you.

### 13.8 Recon

Distribution RECONSTRUCTION: same axis definition as `Distr`, but the shape is
fitted instead of given. The Num weights are free fit variables held beside the
parameter array (so parameter numbering never shifts with Num), regularized by
two Tikhonov terms. Raise D_dif / D_dif2 if the reconstruction oscillates; the
two penalties are the only thing keeping an under-determined shape stable.

### 13.9 Texture and polarization

For thick polarized models, matrix transmission is used:

$$
T(v) = \left[\exp\left(-\hat{\Sigma}(v)\right)\right]_{11}
$$

with source-mode-dependent readout for partial/unpolarized beams:

$$
C_a(v) = \operatorname{tr}\!\left[\exp(-\hat{\Sigma}(v))\,\rho\right]
$$

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

Source modes: SMS is a linearly polarized readout (the polarization degree is
set in *Supp -> Set SMS polarization*, default 0.98); CMS an unpolarized half-trace
readout.
