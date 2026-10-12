# Figures

I organise the figures by analysis stage and pulse family, using the same structure as the [numerical data](../data/README.md). Figures for different simulation parameters are kept together in their pulse-family folder.

## Analysis stages

| Directory | Figures | PNG filename prefix |
| --- | --- | --- |
| [01_baseband](01_baseband/) | Baseband pulse envelopes | `01_baseband_` |
| [02_baseband_fft](02_baseband_fft/) | Baseband FFT spectra | `02_baseband_fft_` |
| [03_iq](03_iq/) | IQ time-domain signals | `03_iq_` |
| [04_iq_fft](04_iq_fft/) | IQ FFT spectra | `04_iq_fft_` |

## Pulse-family folders

The following folders are used within each of the four analysis stages:

| Pulse-family folder | Envelopes | Pulse identifier in PNG filenames |
| --- | --- | --- |
| `rectangular/` | Rectangular | `rect` |
| `gaussian/` | Gaussian | `gaussian` |
| `raised_cosine/` | Hann, Hamming and Blackman | `raised_cosines` |
| `slepian/` | Slepian I and II | `slepian` |
| `kaiser/` | Kaiser | `kaiser` |
| `flat_top_gaussian/` | Flat-Top Gaussian | `flat_top_gaussian` |
| `flat_top_hann/` | Flat-Top Hann | `flat_top_hann` |
| `chebyshev_i/` | Chebyshev I | `chebyshev_I` |
| `chebyshev_ii_wca/` | Chebyshev II WCA | `chebyshev_II_WCA` |
| `gaussian_drag/` | Gaussian DRAG | `drag` |

Hann, Hamming and Blackman are kept together in `raised_cosine/` because they are shown in the same figure. Slepian I and II are similarly kept together in `slepian/`.

## Filenames and numerical data

The [pulse-family scripts](../scripts/pulse_families/README.md) save the figures as PNG files at 300 dpi. Figures are placed here with their original filenames.

The filename identifies the analysis stage, pulse family and simulation parameters. For each calculation, the figure and its corresponding CSV file have matching filenames apart from the extension.

For example, a rectangular baseband pulse with a duration of 100 ns and a sampling rate of 1 GS/s is organised as follows:

| File | Repository path |
| --- | --- |
| Figure | `plots/01_baseband/rectangular/01_baseband_rect_T100ns_fs1.0GSs.png` |
| Numerical data | `data/01_baseband/rectangular/01_baseband_rect_T100ns_fs1.0GSs.csv` |

Figures for different durations, sampling rates and carrier frequencies remain together in the appropriate pulse-family folder.
