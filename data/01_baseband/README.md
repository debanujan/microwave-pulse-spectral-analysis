# Baseband Pulse Data

This directory is for CSV files containing the sampled baseband pulse envelopes. The time coordinate is expressed in nanoseconds; the remaining columns contain the envelope data for the relevant pulse family.

Files in this directory have names beginning with `01_baseband_`.

Example: `rectangular/01_baseband_rect_T100ns_fs1.0GSs.csv`.

See the [data index](../README.md) for the organisation of the numerical outputs.

## Pulse-family folders

Each pulse family has a separate folder. CSV files for different durations, sampling rates and carrier frequencies are kept together in that folder.

| Folder | Pulse family | CSV filename prefix |
| --- | --- | --- |
| [rectangular](rectangular/) | Rectangular | `01_baseband_rect_` |
| [gaussian](gaussian/) | Gaussian | `01_baseband_gaussian_` |
| [raised_cosine](raised_cosine/) | Hann, Hamming and Blackman | `01_baseband_raised_cosines_` |
| [slepian](slepian/) | Slepian I and II | `01_baseband_slepian_` |
| [kaiser](kaiser/) | Kaiser | `01_baseband_kaiser_` |
| [flat_top_gaussian](flat_top_gaussian/) | Flat-Top Gaussian | `01_baseband_flat_top_gaussian_` |
| [flat_top_hann](flat_top_hann/) | Flat-Top Hann | `01_baseband_flat_top_hann_` |
| [chebyshev_i](chebyshev_i/) | Chebyshev I | `01_baseband_chebyshev_I_` |
| [chebyshev_ii_wca](chebyshev_ii_wca/) | Chebyshev II WCA | `01_baseband_chebyshev_II_WCA_` |
| [gaussian_drag](gaussian_drag/) | Gaussian DRAG | `01_baseband_drag_` |

The `raised_cosine/` folder contains the joint Hann, Hamming and Blackman CSV files. The `slepian/` folder contains the joint Slepian I and II CSV files.
