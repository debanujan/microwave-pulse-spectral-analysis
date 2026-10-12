# Baseband Spectral Data

This directory is for CSV files containing the FFT spectra of the baseband pulse envelopes. Frequencies are expressed in gigahertz and FFT magnitudes in decibels, with each spectrum normalised to a maximum of 0 dB.

Files in this directory have names beginning with `02_baseband_fft_`.

See the [data index](../README.md) for the organisation of the numerical outputs.

## Pulse-family folders

Each pulse family has a separate folder. CSV files for different durations, sampling rates and carrier frequencies are kept together in that folder.

| Folder | Pulse family | CSV filename prefix |
| --- | --- | --- |
| [rectangular](rectangular/) | Rectangular | `02_baseband_fft_rect_` |
| [gaussian](gaussian/) | Gaussian | `02_baseband_fft_gaussian_` |
| [raised_cosine](raised_cosine/) | Hann, Hamming and Blackman | `02_baseband_fft_raised_cosines_` |
| [slepian](slepian/) | Slepian I and II | `02_baseband_fft_slepian_` |
| [kaiser](kaiser/) | Kaiser | `02_baseband_fft_kaiser_` |
| [flat_top_gaussian](flat_top_gaussian/) | Flat-Top Gaussian | `02_baseband_fft_flat_top_gaussian_` |
| [flat_top_hann](flat_top_hann/) | Flat-Top Hann | `02_baseband_fft_flat_top_hann_` |
| [chebyshev_i](chebyshev_i/) | Chebyshev I | `02_baseband_fft_chebyshev_I_` |
| [chebyshev_ii_wca](chebyshev_ii_wca/) | Chebyshev II WCA | `02_baseband_fft_chebyshev_II_WCA_` |
| [gaussian_drag](gaussian_drag/) | Gaussian DRAG | `02_baseband_fft_drag_` |

The `raised_cosine/` folder contains the joint Hann, Hamming and Blackman CSV files. The `slepian/` folder contains the joint Slepian I and II CSV files.
