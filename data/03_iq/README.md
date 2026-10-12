# IQ Time-Domain Data

This directory is for CSV files containing the time-domain data for the complex IQ-modulated signals. The time coordinate is expressed in nanoseconds, and the signal columns include the in-phase and quadrature components for the relevant pulse family.

Files in this directory have names beginning with `03_iq_`. The filenames include the carrier frequency.

See the [data index](../README.md) for the organisation of the numerical outputs.

## Pulse-family folders

Each pulse family has a separate folder. CSV files for different durations, sampling rates and carrier frequencies are kept together in that folder.

| Folder | Pulse family | CSV filename prefix |
| --- | --- | --- |
| [rectangular](rectangular/) | Rectangular | `03_iq_rect_` |
| [gaussian](gaussian/) | Gaussian | `03_iq_gaussian_` |
| [raised_cosine](raised_cosine/) | Hann, Hamming and Blackman | `03_iq_raised_cosines_` |
| [slepian](slepian/) | Slepian I and II | `03_iq_slepian_` |
| [kaiser](kaiser/) | Kaiser | `03_iq_kaiser_` |
| [flat_top_gaussian](flat_top_gaussian/) | Flat-Top Gaussian | `03_iq_flat_top_gaussian_` |
| [flat_top_hann](flat_top_hann/) | Flat-Top Hann | `03_iq_flat_top_hann_` |
| [chebyshev_i](chebyshev_i/) | Chebyshev I | `03_iq_chebyshev_I_` |
| [chebyshev_ii_wca](chebyshev_ii_wca/) | Chebyshev II WCA | `03_iq_chebyshev_II_WCA_` |
| [gaussian_drag](gaussian_drag/) | Gaussian DRAG | `03_iq_drag_` |

The `raised_cosine/` folder contains the joint Hann, Hamming and Blackman CSV files. The `slepian/` folder contains the joint Slepian I and II CSV files.
