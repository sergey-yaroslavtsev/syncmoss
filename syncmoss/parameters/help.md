# SYNCmoss help

Short reminders of what the interface does not show. Details, under the same
section numbers: *Supp -> Help (extended)*. Every model's parameters:
*Supp -> Models description*. **Ctrl+F** finds a word.

---

## 1. Simulation without a spectrum

`Model_6` in the path box + **Show model**: the model on 4096 points over
±6 mm/s, no data (`Model_6.5`, `model_6` work too; any other text is a path).

## 2. Linking parameters

| Typed in a value field | Means | Field |
| --- | --- | --- |
| `=[X,Y]` | follows parameter X: `p[X] * Y` | darkorange |
| `=(X)` | own value per spectrum in a one-model fit, from X; elsewhere just X | lightgreen |
| (emptied) | its link pointed at a deleted parameter: fill it in | red |

- Right-click a value: *link to another parameter*, *make it independent*.
- Links and `p[i]` follow inserted/deleted rows. Write `p[12]` exactly (no
  spaces).
- A linked parameter is not fitted; a link is how components share a value.

## 3. Finding a parameter number

Press and hold a parameter's name: it shows `p[12]`. Numbers start at the
baseline (`p[0]` … `p[7]`) and shift when rows are added or removed.

## 4. Expression, Distr, Corr, Recon texts

| Row | Text | Example |
| --- | --- | --- |
| `Expression` | formula in `p[i]`; its value is a parameter of its own | `sqrt(p[3]**2+p[5]**2)` |
| `Distr` | density in `X`, from `L` to `R` in `Num` steps | `exp(-X**2/2)` |
| `Corr` | a second parameter as a function of `X` | `0.3*X` |
| `Recon` | the shape is fitted; `D_dif`, `D_dif2` (0…1) smooth it | `weights` empty = uniform start |

- Functions: `exp`, `log`, `sqrt`, `sin`, `arctan`, `abs`, … by name, any other
  NumPy one as `np.<name>`.
- `N` = the spectrum's place in the path box, `N1`, `N2`, … = its own values
  (section 7.7).
- `par` counts within the row above (`1` = its second parameter, `δ` mostly);
  the driven parameter gets a grey frame. `T` cannot be distributed.
- `Distr`/`Recon` directly under a component, `Corr` under a
  `Distr`/`Corr`/`Recon`; a different `par` for each.
- Nothing may read a driven parameter or a text column: put shared values in
  `Variables`.
- Expressions are calculated from the top: use only Expressions above.
- *Load expression* (row menu): ready-made expressions, `{X}` filled in.

## 5. Keyboard and mouse

| Action | Shortcut |
| --- | --- |
| Fit | `F5` |
| Show model | `Enter` |
| Take result as new model | `F8` |
| Zoom the plot at the cursor | mouse wheel over the figure |
| Apply typed exclusion regions | `Enter` in their box |
| Cancel *Pick on plot* | `Esc` |
| Clean model (baseline kept) | double-click *Clean model* |
| Replot a stored fit | its row button in the results table |
| Find a word in a help window | `Ctrl+F`, then `Enter` / `Shift+Enter` |

## 6. Parameters table details

- **fix** freezes a parameter; a greyed box with a lock is never fitted.
- Small fields under a value: lower / upper bound (empty = none).
- `Onr`, `c²nr`, `linnr` start linked to `Os`, `c²s`, `lins`; both polynomials
  are centred on `Os` / `Onr`. CMS on: `Nnr` = 0 becomes `=[0,0.67]`.
- `Nbaseline` (silver row) opens the next spectrum; `Layer` separates the layers
  of a stack.
- `Library` / `Load model` ADD a model at that row; the baseline is kept.
- `Spectrum`: a measured `.dat` as a component, fitted with lines first (CMS:
  Nnr/Ns fitted in 0.1…1.2; SMS: 0); free `T`, `δ`, `G`. For comparing spectra
  and preliminary fits, not final results (extended help, 13.10).
- *Supp*: theme (first entry), Be / KB impurity presets (pick the entry again
  to load new values).

## 7. Spectrum files and instrumental function

### 7.1 The instrumental function in .dat files

Header lines `#@GCMS G Δθ d D` (CMS) or `#@INSexp` / `#@INSth` with `#@INSint`
(SMS), written by RAW → .dat. *Instrumental function → Switch to the global
instrumental function* ignores them; *Overwrite instrumental function in
selected .dat files* rewrites them.

### 7.2 The CMS source: G, Δθ, d and D

- **G**: Gaussian FWHM of the source line, mm/s. **Δθ**: cosine smearing, rad
  (0 = off). SMS ignores both.
- Calibrate with the Δθ you fit with. Cyan line wiggles at large Δθ: raise JN.
- *Supp → Set CMS source and aperture diameters*: d/D (default 5 / 20 mm).

### 7.3 The Instrumental function button

| Entry | Does |
| --- | --- |
| *Refine: …* | continues from the stored function (usually the better choice) |
| *Find: …* | restarts from the built-in values |
| *single line* / *pure α-Fe* / *model* | reference: Singlet, α-Fe, or the table's model |
| *Refine Δθ: …* (CMS) | fits Δθ, G held |
| *Calibrate + find G, Δθ* (CMS) | raw α-Fe + raw single line in one fit |
| *Save G, Δθ, d and D as defaults* | start values of the next session |

Every search saves its result. *model* with several spectra: one function for
all; `Recon` refused.

### 7.4 SMS: theoretical or Gaussian instrumental function

*Supp → Choose how to approximate instrumental function*: **Theory** (simulated
⁵⁷FeBO₃ source, physical, slower) or **Set of Gaussians** (default).
*Supp → Plot instrumental function …* shows the one in use.

### 7.5 The calibration file

First line `# <method> <n1> <n2>`: `sin` or `lin` folding, raw-channel range.

### 7.6 Several spectra

A list in the path box (`a.dat, b.dat`); *Multispectra settings* chooses:

| Mode | Fit |
| --- | --- |
| sequence | each spectrum alone, from the initial guess or the previous result |
| simultaneous, one model | all at once, shared values; `=(X)` per spectrum |
| (`Nbaseline` rows) | all at once, as built: one row per extra spectrum |

*Fit one spectrum…* fits one of them, saves nothing. A **result window** shows
every spectrum after the fit.

### 7.7 Parameters of the spectra

`('Fe_4K.dat', 4.2, 0)` gives that spectrum `N1 = 4.2`, `N2 = 0`.
*Multispectra settings → Load / Save parameters of the spectra*: a text file
(`#basename`, `#number`, one line per parameter).

### 7.8 Other spectrum sources

A folder (`path/`): its first `.dat`/`.mca`. `McaAcq_channel_<…>`: a Bliss
channel (Linux, `bliss` installed).

### 7.9 Channel average

*Supp → Average model over each channel* (ON by default): every point is the
model averaged over the velocities the drive runs through while it is counted.
Find the instrumental function and calibrate with the setting you fit with.

## 8. Exclusion regions

**Apply exclusion** (plot toolbar), then `lo:hi; lo:hi` in mm/s + **Enter**, or
**Pick on plot** and click both ends (**Esc** cancels). Excluded points are not
fitted. Off at every start; a change voids the fit on screen. For good:
*Change spectrum(a) → Save spectrum without excluded points* (their counts
become `nan`; the velocities stay).

## 9. Errors, χ² and the correlation matrix

- Fits maximise the Poisson likelihood; the plot shows **χ²** per degree of
  freedom, ±√(2/ν) in brackets. D and the rest: tab *Fit quality* (9.1).
- Errors = statistical 1σ × √χ² (× 1 when χ² = 0: a spectrum without noise;
  *Supp → Empirical correction of errors by χ²* turns the factor off). A
  parameter with no error is redundant (see the log).

### 9.1 Fit quality

Third tab of the results: each number with its ideal and its 1σ (its spread
for a right model and Poisson counting). Within 2σ nothing is seen; beyond 3σ
it is real. Each S comes with the F it divides χ² by (F = 1: Poisson).

| quantity | ideal | says |
|---|---|---|
| χ² | 1 | model and counting together (Poisson counts) |
| D | ≈ 1 | the deviance the fit minimised |
| S (fit) | 1 | look first: above 1 the model misses something, whatever the counting |
| S (no model) | 1 | χ² over the counting noise of the counts alone; its F: is the counting Poisson? |
| S (SG) | 1 | the same with another smooth: a cross-check, holds at very high counts |
| MISFIT, % | 0 | size of the misfit against the absorption; √ = R-factor |

S (fit) reads low for misfits narrower than ~3 channels, high for data
interpolated or smoothed elsewhere. S (no model) can read low (its F high) with
deep natural lines at ≳10⁷ counts per point: then trust S (SG). Neither is
shown when the natural line spans fewer than 2 channels.

## 10. Saving

**Save result** writes `<base>_param.txt`, `_graf.txt`, `_combo.png`, `.svg`,
`_result_model.mdl` (+ `_distributions.png`, `_inputs.txt`, `_exclusion.txt`
when used). A sequence or one-model fit writes ONE `_param.txt` and ONE
`_result_table_PNG.html`, extended by every run.

## 11. Other

- **! INTERRUPT !** restarts the worker pool; model and spectrum are kept.
- *Supp → Set SMS polarization*: SMS polarization degree (default 0.98).
- *Supp → Set number of points for full transmission integral*: 32 SMS, 64 CMS;
  a flat cyan *Integration check (×4)* line means enough.
- *Supp → Save settings as default*: the next starts keep the current settings
  (`config.json` in the parameters folder); *Restore default settings* deletes it.
- *Change spectrum(a)*: sum, subtract model, spectrum from model, half points,
  without excluded points.
- **Distribution / Spectrum** shows the `Distr`/`Recon` curves.

## 12. When something goes wrong

The *report a bug* window holds the error: send it with what you did, the
spectrum and the `.mdl` (*Supp → Contact the author*).
