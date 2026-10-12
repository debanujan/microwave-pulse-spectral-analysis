# Original pulse-family generators

These ten standalone scripts export the complete first calculation cell from
the original saved Colab notebooks. Pulse functions, sweep settings, FFT
normalization, CSV contents, filenames, and plot styles are preserved.
Only a source-reference header has been added. Colab's later ZIP, directory-listing,
and browser-download cells are omitted from the Python exports.

## Script index

| Script | Pulse family and saved shape settings | Original notebook |
| --- | --- | --- |
| [rectangular.py](rectangular.py) | Rectangular pulse | [Rectangular Pulse.ipynb](https://colab.research.google.com/drive/17u10JDR9hADYfyZCW6Wh6OwheQ7FfOcA) |
| [gaussian.py](gaussian.py) | Gaussian, with nominal duration `T = 2 sigma` | [Gaussian Pulse.ipynb](https://colab.research.google.com/drive/1o8Gv4hRi4rnpGoqGy4nzVpd__lAN7LT4) |
| [raised_cosine.py](raised_cosine.py) | Hann, Hamming, and Blackman windows | [Raised Cosine Pulse.ipynb](https://colab.research.google.com/drive/1tVg5Kj73dQ9wEVl7J6xM20ZXLuvk8iGz) |
| [slepian.py](slepian.py) | Zeroth and first DPSS tapers, `NW = 3` | [Slepian Pulse.ipynb](https://colab.research.google.com/drive/1nuNJHEh42kDCJi7Juz1-ramxqUjvXa6y) |
| [kaiser.py](kaiser.py) | Kaiser window, `beta = 8` | [Kaiser.ipynb](https://colab.research.google.com/drive/1Wv2p77eIAzAhfXNoqMre9hO0ptyvprfn) |
| [flat_top_gaussian.py](flat_top_gaussian.py) | Baseline-corrected Gaussian edges, flat fraction `0.5`, edge factor `3` | [Flat_top_Gaussian.ipynb](https://colab.research.google.com/drive/15_VvPMn8zJ8LqJ-jnuPXlCtNFm41uaBL) |
| [flat_top_hann.py](flat_top_hann.py) | Hann rise and fall, flat fraction `0.5` | [Flat-top-cosine.ipynb](https://colab.research.google.com/drive/1lWOJO38H8qh9ycYDSnVSgGZdCODKQt2I) |
| [chebyshev_i.py](chebyshev_i.py) | Symmetric Dolph-Chebyshev windows, attenuation settings `60` and `80` dB | [Chebyshev_I.ipynb](https://colab.research.google.com/drive/12zkQry2_s07jnp9MD8ej7Vn-CNtXSOCZ) |
| [chebyshev_ii_wca.py](chebyshev_ii_wca.py) | Antisymmetric WCA/Remez construction, attenuation setting `60` dB | [Chebyshev_II_WCA.ipynb](https://colab.research.google.com/drive/1woWeTZfcGfAobfoUIKE9uOO5i4hhQJVY) |
| [gaussian_drag.py](gaussian_drag.py) | Endpoint-corrected Gaussian DRAG, `beta = 1`, `alpha_Hz = -250e6`, `SIGMA_DIVISOR = 2` | [DRAG.ipynb](https://colab.research.google.com/drive/1UT6tXylIQi5m6u6FHNXYLBUrOyXSz8mO) |

## Run a script

Use Python 3.11 or newer with the provided dependency versions. From the
repository root:

```bash
python -m pip install -r requirements.txt
python scripts/pulse_families/rectangular.py
```

Replace `rectangular.py` with any filename in the table. These are standalone
batch scripts: running or importing a file executes its parameter sweep.

The default controls at the beginning of each script are:

```python
SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_DATA = True
```

Plots are saved as PNG files at 300 dpi, and the numerical arrays used for the
plots are saved as CSV files with column headers.

## Small first run

The saved full sweeps use six durations, six sampling rates, and three carrier
frequencies. For a small run, change the following settings near the beginning
of the chosen script:

```python
durations = np.array([100])          # nanoseconds
fs_list = np.array([1]) * 1e9        # samples per second: 1 GS/s
fc_list = np.array([50]) * 1e6       # carrier frequency: 50 MHz
```

This produces one output pair for each of the four stages. The original full
sweep settings are:

| Setting | Saved values |
| --- | --- |
| Duration | 50, 100, 250, 500, 750, 1000 ns |
| Sampling rate | 0.5, 1, 2, 4, 6, 10 GS/s |
| Carrier frequency | 50, 100, 150 MHz |

Shape parameters and time-axis limits are specified in each script.

## Output organization

The `main_folder` setting names the output study folder. Its `data` and `plots`
subfolders are created relative to the working directory from which Python is
launched. For the rectangular generator run from the repository root:

| Folder | Contents |
| --- | --- |
| `rectangular_pulse_study/data/` | CSV numerical data |
| `rectangular_pulse_study/plots/` | PNG plots |

The output filename prefixes identify the analysis stage:

| Prefix | Analysis |
| --- | --- |
| `01_baseband` | Baseband pulse versus time |
| `02_baseband_fft` | Baseband spectrum |
| `03_iq` | IQ signal versus time |
| `04_iq_fft` | IQ spectrum |

Running the same settings again replaces files with matching names. Generated
study folders in the repository root are excluded by `.gitignore`; curated data
and figures can be organized separately.

## Conventions retained from the notebooks

- The saved generators use amplitude or peak scaling with `A = 1` (or `1.0`).
  Spectra are individually normalized to a 0 dB maximum. Equal-area strength
  calibration and comparative metrics require a subsequent analysis step.
- In `gaussian.py`, `T = 2 sigma` denotes a nominal interval. The Gaussian is
  evaluated over the full saved time axis, including its tails.
- Slepian II is the first DPSS taper and has a sign change. The Chebyshev II
  WCA construction is also signed and antisymmetric.
- The Chebyshev II script preserves the notebook's Remez retries and prototype
  interpolation for large sample counts. Its attenuation parameter sets design
  weights; measured attenuation depends on the resulting pulse.
- DRAG uses the notebook's complex envelope and sign convention,
  `Q(t) = -beta * dI(t)/dt / (2*pi*alpha_Hz)`.

## Verification

All exports were checked against their source calculation cells and compiled
successfully. Each script was run with a reduced sweep of 50 ns, 0.5 GS/s, and
50 MHz, retaining its original shape parameters and plot/data saving. Every run
produced all four CSV files and all four PNG files. CSV arrays were checked for
finite values and increasing time/frequency coordinates. The Chebyshev II
prototype-interpolation branch was also exercised at 2500 samples.

Checks used NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8, as recorded in
`requirements.txt`. The complete default sweeps were not rerun during export.
