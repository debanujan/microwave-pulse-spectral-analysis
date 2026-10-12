# microwave-pulse-spectral-analysis
Codes, data, and plots for time- and frequency-domain analysis of microwave control pulse families for superconducting transmon qubits.

## Pulse-family scripts

The original pulse generators are available in [scripts/pulse_families](scripts/pulse_families/README.md).
Ten standalone Python scripts cover rectangular, Gaussian, Hann/Hamming/Blackman,
Slepian/DPSS, Kaiser, flat-top Gaussian, flat-top Hann, Chebyshev I,
Chebyshev II WCA, and Gaussian DRAG pulses.

Each script generates numerical CSV data and PNG plots for baseband time-domain,
baseband FFT, IQ time-domain, and IQ FFT analysis. The calculation cells retain
the saved source notebooks' pulse definitions, sampling, normalization, and plotting.

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/pulse_families/rectangular.py
```

The default settings run the complete parameter sweep. See the
[script guide](scripts/pulse_families/README.md) for a small first run, output paths,
shape conventions, and source-notebook links.

