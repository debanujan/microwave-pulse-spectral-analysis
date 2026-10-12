# Pulse Families and Spectral Analysis

These scripts form the numerical part of my M.Sc. thesis, **Spectral Analysis of Microwave Pulses for Superconducting Qubit Control**. I use them to examine how pulse shape, duration and sampling rate affect the frequency content of a control signal, and how its spectrum changes after IQ modulation.

The calculations cover rectangular, Gaussian, generalised cosine, Slepian, Kaiser, flat-top, Chebyshev and Gaussian DRAG envelopes. Each script generates the time-domain signals, calculates their spectra, and saves the numerical data and corresponding figures.

## Pulse families

| Python script | Pulse family and parameters | Colab notebook |
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

## Numerical procedure

For each pulse family, I carry out the analysis in four stages:

1. **Baseband pulse:** generate the sampled envelope over the selected time interval.
2. **Baseband spectrum:** calculate the FFT of the envelope and express its magnitude in decibels.
3. **IQ signal:** multiply the envelope by a complex carrier and obtain the in-phase and quadrature components.
4. **IQ spectrum:** calculate the FFT of the modulated signal to examine the frequency translation and spectral profile.

For an envelope $\Omega(t)$ and carrier frequency $f_c$, the complex modulated signal is

$$
s_{\mathrm{IQ}}(t)=\Omega(t)e^{i2\pi f_c t}.
$$

For the real-valued pulse families, $\Omega(t)=w(t)$. For Gaussian DRAG, the envelope is complex:

$$
\Omega(t)=I(t)+iQ(t),
\qquad
Q(t)=-\frac{\beta}{2\pi\alpha_{\mathrm{Hz}}}\frac{dI(t)}{dt}.
$$

The DRAG calculation uses $\alpha_{\mathrm{Hz}}=-250\,\mathrm{MHz}$ and $\beta=1$. The in-phase envelope is an endpoint-corrected Gaussian.

The scripts use an amplitude parameter $A=1$, with the scaling defined for each pulse family. For the spectral plots, I normalise each FFT magnitude to a maximum of 0 dB. This representation allows comparison of the spectral profiles. Equal-area calibration and comparative metrics are separate steps in the analysis.

Positive-frequency spectra are used for the real baseband envelopes. The complex DRAG envelope and the IQ signals are analysed using the full shifted spectrum.

## Simulation parameters

The parameter sweep uses the following values:

| Parameter | Values |
| --- | --- |
| Pulse duration | 50, 100, 250, 500, 750 and 1000 ns |
| Sampling rate | 0.5, 1, 2, 4, 6 and 10 GS/s |
| Carrier frequency | 50, 100 and 150 MHz |

Time is expressed in seconds within the calculations. The saved time-domain data use nanoseconds, and the saved spectral data use gigahertz. Shape parameters and observation intervals are specified within the individual scripts.

For the Gaussian family, the nominal duration is defined as $T=2\sigma$. The Gaussian is evaluated over the full observation interval, including its tails. The flat-top envelopes use a flat section occupying half of the total pulse duration.

Slepian I and Slepian II correspond to the zeroth and first DPSS tapers, respectively. Slepian II has a sign change. The Chebyshev II WCA pulse is also signed and antisymmetric; it is constructed using a Remez design, with prototype interpolation for large sample counts. Its attenuation parameter controls the design weights, while the resulting attenuation is determined from the calculated spectrum.

## Running the calculations

The scripts use NumPy, SciPy and Matplotlib. The dependency versions are listed in the repository's `requirements.txt` and require Python 3.11 or newer.

From the repository root, install the dependencies and run the required script:

```bash
python -m pip install -r requirements.txt
python scripts/pulse_families/rectangular.py
```

The filename can be replaced with any script listed above. These files are standalone batch programs; importing a file also executes its configured parameter sweep.

The following controls appear at the beginning of each script:

```python
SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_DATA = True
```

With these settings, the scripts save the data and figures without displaying each plot. Figures are saved as PNG files at 300 dpi, and numerical data are saved as CSV files with column headers.

To calculate a single parameter combination, set the sweep arrays to:

```python
durations = np.array([100])       # pulse duration in ns
fs_list = np.array([1]) * 1e9     # sampling rate: 1 GS/s
fc_list = np.array([50]) * 1e6    # carrier frequency: 50 MHz
```

This generates a data file and a figure for each of the four stages.

## Data and figures

The `main_folder` variable specifies the study directory. Within this directory, numerical data are saved in `data/` and figures in `plots/`. Output paths are relative to the working directory from which the script is run.

For example, the rectangular pulse script creates:

| Output directory | Contents |
| --- | --- |
| `rectangular_pulse_study/data/` | Numerical data in CSV format |
| `rectangular_pulse_study/plots/` | Figures in PNG format |

The filenames identify the analysis stage and the relevant simulation parameters:

| Filename prefix | Analysis stage |
| --- | --- |
| `01_baseband` | Baseband pulse |
| `02_baseband_fft` | Baseband spectrum |
| `03_iq` | IQ signal |
| `04_iq_fft` | IQ spectrum |

Repeating a calculation with the same settings replaces files with matching names. Generated study directories in the repository root are excluded by `.gitignore`; selected data and figures can be added separately.
