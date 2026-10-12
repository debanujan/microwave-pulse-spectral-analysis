# Numerical Data

I organise the numerical outputs by the four stages of the pulse-family analysis. Each stage contains CSV files for the different pulse families and simulation parameters.

| Directory | Numerical output | CSV filename prefix |
| --- | --- | --- |
| [01_baseband](01_baseband/) | Baseband pulse envelopes versus time | `01_baseband_` |
| [02_baseband_fft](02_baseband_fft/) | Baseband FFT spectra | `02_baseband_fft_` |
| [03_iq](03_iq/) | IQ signals versus time | `03_iq_` |
| [04_iq_fft](04_iq_fft/) | IQ FFT spectra | `04_iq_fft_` |

The CSV files retain the filenames and column headers produced by the [pulse-family scripts](../scripts/pulse_families/README.md). The filenames identify the pulse family, pulse duration, sampling rate and, where applicable, carrier frequency and shape parameters.

Time-domain data use nanoseconds for the time coordinate. Spectral data use gigahertz for the frequency coordinate and decibels for the relative FFT magnitude. The spectral magnitudes are normalised to a maximum of 0 dB.

For example, a rectangular pulse with a duration of 100 ns and a sampling rate of 1 GS/s has a baseband data file named `01_baseband_rectangular_T100ns_fs1.0GSs.csv`. This file belongs in `01_baseband/`.

Each CSV file is placed in the directory corresponding to its filename prefix. The original names and numerical values are retained.
