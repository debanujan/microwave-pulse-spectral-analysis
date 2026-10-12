# IQ Spectral Data

This directory is for CSV files containing the FFT spectra of the complex IQ-modulated signals. Frequencies are expressed in gigahertz and FFT magnitudes in decibels, with each spectrum normalised to a maximum of 0 dB.

Files in this directory have names beginning with `04_iq_fft_`. The filenames include the carrier frequency.

See the [data index](../README.md) for the organisation of the numerical outputs.

## Pulse-family folders

Each pulse family has a separate folder. CSV files for different durations, sampling rates and carrier frequencies are kept together in that folder.

| Folder | Pulse family | CSV filename prefix |
| --- | --- | --- |
| [rectangular](rectangular/) | Rectangular | `04_iq_fft_rect_` |
| [gaussian](gaussian/) | Gaussian | `04_iq_fft_gaussian_` |
| [raised_cosine](raised_cosine/) | Hann, Hamming and Blackman | `04_iq_fft_raised_cosines_` |
| [slepian](slepian/) | Slepian I and II | `04_iq_fft_slepian_` |
| [kaiser](kaiser/) | Kaiser | `04_iq_fft_kaiser_` |
| [flat_top_gaussian](flat_top_gaussian/) | Flat-Top Gaussian | `04_iq_fft_flat_top_gaussian_` |
| [flat_top_hann](flat_top_hann/) | Flat-Top Hann | `04_iq_fft_flat_top_hann_` |
| [chebyshev_i](chebyshev_i/) | Chebyshev I | `04_iq_fft_chebyshev_I_` |
| [chebyshev_ii_wca](chebyshev_ii_wca/) | Chebyshev II WCA | `04_iq_fft_chebyshev_II_WCA_` |
| [gaussian_drag](gaussian_drag/) | Gaussian DRAG | `04_iq_fft_drag_` |

The `raised_cosine/` folder contains the joint Hann, Hamming and Blackman CSV files. The `slepian/` folder contains the joint Slepian I and II CSV files.
