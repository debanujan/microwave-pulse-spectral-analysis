# Numerical Data

I organise the numerical outputs first by analysis stage and then by pulse family. Within each pulse-family folder, the CSV files cover the different pulse durations, sampling rates and carrier frequencies.

| Directory | Numerical output | CSV filename prefix |
| --- | --- | --- |
| [01_baseband](01_baseband/) | Baseband pulse envelopes versus time | `01_baseband_` |
| [02_baseband_fft](02_baseband_fft/) | Baseband FFT spectra | `02_baseband_fft_` |
| [03_iq](03_iq/) | IQ signals versus time | `03_iq_` |
| [04_iq_fft](04_iq_fft/) | IQ FFT spectra | `04_iq_fft_` |

## Pulse-family folders

The following folders are used within each of the four analysis-stage directories:

| Pulse-family folder | Envelopes | Pulse identifier in CSV filenames |
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

Hann, Hamming and Blackman are kept together in `raised_cosine/` because the generator saves their data in separate columns of the same CSV. Slepian I and II are similarly kept together in `slepian/`.

Files for different simulation parameters remain together within each pulse-family folder. The parameters are identified by the filenames.

## Filenames and units

The CSV files retain the filenames and column headers produced by the [pulse-family scripts](../scripts/pulse_families/README.md). The filenames identify the pulse family, pulse duration, sampling rate and, where applicable, carrier frequency and shape parameters.

Time-domain data use nanoseconds for the time coordinate. Spectral data use gigahertz for the frequency coordinate and decibels for the relative FFT magnitude. The spectral magnitudes are normalised to a maximum of 0 dB.

For example, a rectangular pulse with a duration of 100 ns and a sampling rate of 1 GS/s has a baseband data file named `01_baseband_rect_T100ns_fs1.0GSs.csv`. This file belongs in `01_baseband/rectangular/`.

Each CSV file is placed in the directory corresponding to its analysis stage and pulse family. The original names and numerical values are retained.
