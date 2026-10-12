# Spectral Analysis of Microwave Pulses for Superconducting Qubit Control

This repository documents the numerical work associated with my M.Sc. thesis, **Spectral Analysis of Microwave Pulses for Superconducting Qubit Control**.

I study the time-domain profiles and frequency-domain characteristics of microwave control-pulse envelopes relevant to superconducting transmon qubits. The calculations examine the dependence of the spectra on pulse shape, duration and sampling rate, together with the frequency translation produced by IQ modulation.

## Pulse-family scripts

The [pulse-family directory](scripts/pulse_families/README.md) contains ten standalone Python scripts covering rectangular, Gaussian, Hann, Hamming, Blackman, Slepian/DPSS, Kaiser, flat-top Gaussian, flat-top Hann, Chebyshev I, Chebyshev II WCA and Gaussian DRAG envelopes.

Each script generates numerical data and figures for four stages: the baseband pulse, its FFT, the IQ signal and its FFT. Pulse definitions, shape parameters, normalisation conventions and execution instructions are described in the [pulse-family documentation](scripts/pulse_families/README.md).

## Numerical data

The [data directory](data/README.md) provides separate folders for baseband pulse data, baseband spectra, IQ time-domain data and IQ spectra. CSV files are organised by analysis stage, with the pulse family and simulation parameters retained in their filenames.

## Running the calculations

Use Python 3.11 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/pulse_families/rectangular.py
```

The scripts run the configured parameter sweeps and save CSV data and PNG figures in their respective study directories. The pulse-family documentation explains how to select a single parameter combination and identifies the output files.
