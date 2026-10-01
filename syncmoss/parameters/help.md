# SYNCmoss quick help

Short notes on things the interface does **not** show you. This is not a manual:
buttons that explain themselves are not listed here. For the physics and the
parameter lists of every model see *Supp -> Models description*.

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

- Links are kept in step when you insert or delete model rows: the referenced
  numbers are renumbered for you. The same holds for every `p[i]` inside an
  `Expression`, `Distr` or `Corr` text.
- If the parameter a link pointed at is itself deleted, the field is emptied and
  turned **red**. Show model and Fit then refuse to start until you fill it in.
- The link is applied both when showing the model and inside every fit
  iteration, so the linked parameter never has its own degree of freedom.
- A link is also how a parameter is shared between components — e.g. one common
  central shift for two sextets.

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
  `33.0-p[12]`. Bare NumPy functions (`sqrt`, `exp`, `log`, `sin`, `abs`, …) are
  available. The result becomes a parameter of its own — click its label to read
  the number, then point real parameters at it with `=[X,Y]`. That is how a
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

Every one of these texts may also use the **parameters of the spectrum** (see
section 7): `N` is the spectrum's number in the sequence — its position in the
path box, counted from 1 — and `N1`, `N2`, … are the numbers given with it
(a temperature, an angle, …). So `p[9]+N*2` or `p[9]+0.01*N1`, linked into a
parameter with `=[X,1]`, forces that parameter to follow the spectrum number or
the temperature through a sequence. A single fit and **Show model** use the
first spectrum (`N` = 1). A value a spectrum does not have blocks the start with
a message, and the names are refused in a model with `Nbaseline`.

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
| Clean model (baseline kept) | double-click the *Clean model* button |
| Replot a stored fit | click its row button in the results table |

## 6. Parameters table details

- The **fix** check box next to each name freezes that parameter. Orientation
  angles, texture parameters and the like start fixed on purpose — untick them
  only when the data can support them.
- A check box that is **greyed out with a lock icon** is structural and can never
  be fitted: `par`/`Num` of `Distr`, `par` of `Corr`, `par`/`Num`/`D_dif`/`D_dif2`
  of `Recon`, `S` of `Relax_MS`, `Num` of `ASM`, `N/Γ` of `SCDW`.
- The two small fields under each value are the **lower and upper bound**. Leave
  a bound empty for "unbounded".
- Empty rows between models are harmless — they are dropped when the model is
  saved.
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

## 7. Spectrum files and instrumental function

- A `.dat` file may carry the instrumental function in its header:
  `#@GCMS` marks a CMS spectrum (and its Gaussian width), `#@INSexp` +
  `#@INSint` an SMS one. SYNCmoss writes those lines when it converts a
  spectrum, and reads them back per spectrum — so a batch may mix CMS and SMS
  files. Switch the behaviour with *Instrumental function -> do not use
  instrumental function from .dat file*.
- **Find Instr. func. NEW** fits the *theoretical* SMS instrumental function
  instead of a free sum of Gaussians: the simulated energy distribution of a
  ⁵⁷FeBO₃ synchrotron Mössbauer source, with four physical numbers —
  the rocking-curve position θ [µrad], the staggered hyperfine field B_s [T]
  (the temperature knob), the quadrupole splitting ΔE_Q [mm/s] and the source
  shift [mm/s]. Same three reference absorbers as the ordinary search. It is
  slower (≈100 s against ≈15 s) and, having 4 free shape numbers instead of 9,
  will usually give a slightly *higher* χ² — what it gives back is parameters
  that mean something and the correct v⁻⁴ line wings, which a Gaussian sum
  cannot have at all.
- The theoretical result is stored in `parameters/INSacc.txt` and, in converted
  `.dat` files, in an extra `#@INSacc` header line. `#@INSexp`/`#@INSint` are
  still written next to it, now holding the best Gaussian-sum stand-in for the
  same source, so nothing that only understands the old lines is left without an
  instrumental function. While `INSacc.txt` exists it is the one every fit uses;
  running the ordinary search again, or *Reset to default values*, removes it.
- *Supp -> Plot instrumental function from memory / from spectrum* draws what is
  actually in use — from `INSacc.txt`/`INSexp.txt`, or from the loaded
  spectrum's own header — with its FWHM, centre and first moment, on a linear
  and a logarithmic scale. The log panel is where a theoretical and an empirical
  instrumental function stop looking alike. It opens in its **own window**, with
  its own zoom toolbar and a Save button, and stays there while you work on the
  spectrum; the main plot is never touched.
- The calibration file's first line is `# <method> <n1> <n2>`: `sin` or `lin`
  folding and the raw-channel range. Without it, `sin` folding over all channels
  is assumed.
- Several files in the path box (a Python-style list, or just comma-separated)
  mean **sequence** fitting — unless the model contains `Nbaseline` rows, in
  which case they are fitted **simultaneously** (one `Nbaseline` per extra
  spectrum).
- A spectrum in the path box may carry parameters of its own, used as `N1`,
  `N2`, … in the formulas (section 4): write it as a tuple,
  `[('Fe_4K.dat', 4.2, 0), ('Fe_77K.dat', 77, 0)]`. Plain paths and tuples may
  be mixed.
- *Sequence Fitting → Load parameters of the spectra* fills those tuples in from
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

- A path ending in `/` or `\` is read as a folder: the first `.dat`/`.mca` in it
  is used.
- On Linux with the `bliss` package installed (it is not a dependency), a Bliss
  channel name `McaAcq_channel_<...>` can stand in for a file: the spectrum the
  MCA is accumulating is read from Bliss and folded like an `.mca` file with the
  current calibration, afresh on every **Show spectrum**, **Show model** and
  **Fit**. The beacon server is taken from the `BEACON_HOST` environment
  variable, or is `id14:25000` when that is unset.

## 8. Saving

**Save result** writes six or seven files next to the save path, not one. `<base>` is the
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
  written. (Sequence fitting skips this — it would ask once per spectrum.)

A **sequence** saves as it goes, without asking: ONE `<base>_param.txt` for the
whole run — a line of names, then one line per fitted spectrum — and ONE
`<base>_result_table_PNG.html` with the pictures of every spectrum (figure +
table, and the distributions when there are any). Both are started afresh by
each run. The curves, figure and pictures of each spectrum are also saved next
to them under the spectrum's own name (`Fe_4K_graf.txt`, `Fe_4K_combo.png`, …).

## 9. Other

- **Interrupt** does not just stop the fit: it terminates the worker pool and
  builds a fresh one. Use it if a calculation is stuck — the fit in progress dies
  with it, but the model and the loaded spectrum are untouched.
- *Supp -> Set polarization* sets the SMS beam linear-polarization degree
  (0 = unpolarized, 1 = fully polarized; default 0.98). It changes the readout of
  every thick model.
- *Supp -> Set number of points for full transmission integral* is the
  accuracy↔speed knob of the thickness integral (32 for SMS, 64 for CMS;
  switching the CMS/SMS box moves it between those two defaults for you).
  Whether it is high enough is shown, not guessed: the **cyan
  "Integration check (×4)"** trace under every Show model and Fit is the same
  model recomputed with four times as many points, minus the one you see. A flat
  line means converged; a visible wiggle means raise the number.
- *Change spectrum(a)* holds three destructive-looking but useful operations:
  sum all listed spectra, subtract the current model from the spectrum, and
  halve the number of points.
- Distributions are drawn in the plot area: the **Distribution / Spectrum**
  button toggles the view once a model with `Distr`/`Recon` has been shown.

## 10. When something goes wrong

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
