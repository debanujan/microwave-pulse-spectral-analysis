# Source notebook: Chebyshev_II_WCA.ipynb
# Source: https://colab.research.google.com/drive/1woWeTZfcGfAobfoUIKE9uOO5i4hhQJVY
# Original calculation cell exported without changes.
# Colab-only ZIP and download cells are omitted.

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import remez

# ============================================================
# Plot and Data Control
# ============================================================

SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_DATA = True

main_folder = "chebyshev_II_WCA_remez_60dB_study"
plot_folder = os.path.join(main_folder, "plots")
data_folder = os.path.join(main_folder, "data")

if SAVE_PLOTS:
    os.makedirs(plot_folder, exist_ok=True)

if SAVE_DATA:
    os.makedirs(data_folder, exist_ok=True)


def finish_plot(filename):
    plt.tight_layout()

    if SAVE_PLOTS:
        filepath = os.path.join(plot_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def save_data(filename, header, data_array):
    if SAVE_DATA:
        filepath = os.path.join(data_folder, filename)
        np.savetxt(
            filepath,
            data_array,
            delimiter=",",
            header=header,
            comments=""
        )


# ============================================================
# Parameters
# ============================================================

durations = np.array([50, 100, 250, 500, 750, 1000])
fs_list = np.array([0.5, 1, 2, 4, 6, 10]) * 1e9
fc_list = np.array([50, 100, 150]) * 1e6

# WCA / equiripple attenuation target
# Only -60 dB is used in this version.
attenuation_values = np.array([60])

A = 1

# Same protocol as the earlier 2000 ns pulse-family codes
t_start = 0
t_end = 2000e-9
pulse_start = 500e-9

# Common comparison frequency limit
# Lowest sampling rate = 0.5 GS/s, Nyquist = 0.25 GHz
f_compare = np.min(fs_list) / 2


# ============================================================
# Chebyshev-II WCA / Remez design parameters
# ============================================================
#
# Chebyshev II is implemented as an anti-symmetric equiripple finite impulse
# pulse using the Remez exchange algorithm.
#
# The normalized digital frequency in scipy.signal.remez is in cycles/sample
# when fs = 1.0, so Nyquist is 0.5.
#
# We define a single-mainlobe band and a stopband:
#
#   passband: [epsilon, passband_edge]
#   stopband: [stopband_edge, 0.5 - epsilon]
#
# The passband and stopband edges are scaled with pulse length N, so the
# mainlobe width shrinks for longer pulses.
#
# Increase MAINLOBE_FACTOR or TRANSITION_FACTOR if remez fails for very short
# N or if you want a wider mainlobe.
#

EPSILON = 1e-4
MAINLOBE_FACTOR = 2.0
TRANSITION_FACTOR = 1.0

# For very large N, Remez can be slow. Keep this True for production sweeps.
# The code will still design an anti-symmetric equiripple pulse, but if N is
# larger than MAX_REMEZ_TAPS, it designs a prototype and interpolates it.
#
# Set USE_PROTOTYPE_FOR_LARGE_N = False if you want Remez to run directly
# for every full N, although that may be slow for N ~ 10000.
USE_PROTOTYPE_FOR_LARGE_N = True
MAX_REMEZ_TAPS = 1001


# ============================================================
# FFT Functions
# ============================================================

def compute_fft_positive(signal, Ts):
    n_samples = len(signal)
    fft_output = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n_samples, d=Ts)

    magnitude = np.abs(fft_output) / n_samples
    positive_mask = freqs >= 0

    freqs_pos = freqs[positive_mask]
    magnitude_pos = magnitude[positive_mask]

    magnitude_db_pos = 20 * np.log10(magnitude_pos + 1e-12)
    magnitude_db_pos -= np.max(magnitude_db_pos)

    return freqs_pos, magnitude_db_pos


def compute_fft_complex(signal, Ts):
    n_samples = len(signal)

    fft_output = np.fft.fft(signal)
    fft_output = np.fft.fftshift(fft_output)

    freqs = np.fft.fftfreq(n_samples, d=Ts)
    freqs = np.fft.fftshift(freqs)

    magnitude = np.abs(fft_output) / n_samples

    magnitude_db = 20 * np.log10(magnitude + 1e-12)
    magnitude_db -= np.max(magnitude_db)

    return freqs, magnitude_db


# ============================================================
# WCA / Remez Anti-Symmetric Chebyshev-II Design
# ============================================================

def _make_odd(num):
    """
    For Case-3 anti-symmetric design, use odd length when possible.
    """
    num = int(num)
    if num < 5:
        num = 5
    if num % 2 == 0:
        num += 1
    return num


def _normalize_peak_abs(x, A):
    """
    Normalize signal to peak absolute amplitude A.
    """
    peak = np.max(np.abs(x))
    if peak == 0:
        return x
    return A * x / peak


def design_chebyshev_II_wca_window(N, attenuation_db, A):
    """
    Designs an anti-symmetric Chebyshev-II-type pulse using a WCA/Remez
    equiripple construction.

    This is a concrete WCA design choice:
        - anti-symmetric linear-phase FIR pulse,
        - one low-frequency mainlobe/passband away from zero,
        - equiripple stopband,
        - stopband weight chosen from attenuation_db.

    The result is signed and satisfies anti-symmetric behavior.
    """

    if N < 5:
        raise ValueError("N must be at least 5 for Chebyshev II WCA design.")

    # Direct Remez length or reduced prototype length
    if USE_PROTOTYPE_FOR_LARGE_N and N > MAX_REMEZ_TAPS:
        M = _make_odd(MAX_REMEZ_TAPS)
    else:
        M = _make_odd(N)

    # Define normalized frequency edges in cycles/sample, Nyquist = 0.5
    passband_edge = MAINLOBE_FACTOR / M
    stopband_edge = (MAINLOBE_FACTOR + TRANSITION_FACTOR) / M

    passband_edge = max(passband_edge, 5 * EPSILON)
    stopband_edge = max(stopband_edge, passband_edge + 5 * EPSILON)

    # Keep inside Nyquist
    passband_edge = min(passband_edge, 0.20)
    stopband_edge = min(stopband_edge, 0.45)

    if stopband_edge <= passband_edge:
        stopband_edge = min(passband_edge + 0.05, 0.45)

    bands = [
        EPSILON,
        passband_edge,
        stopband_edge,
        0.5 - EPSILON
    ]

    desired = [1, 0]

    # Stopband weight. Higher attenuation -> stronger stopband suppression.
    stop_weight = 10 ** (attenuation_db / 20)

    # Extremely high weights can make some designs numerically unstable.
    # This cap keeps the sweep robust while still enforcing strong stopband.
    stop_weight = min(stop_weight, 1e4)

    weights_to_try = [
        [1, stop_weight],
        [1, np.sqrt(stop_weight)],
        [1, stop_weight ** 0.25],
        [1, 100]
    ]

    window = None

    for weight in weights_to_try:
        try:
            window = remez(
                numtaps=M,
                bands=bands,
                desired=desired,
                weight=weight,
                type="hilbert",
                fs=1.0,
                maxiter=100,
                grid_density=16
            )
            break
        except Exception:
            window = None

    if window is None:
        raise RuntimeError(
            "Remez design failed. Try increasing MAINLOBE_FACTOR or "
            "TRANSITION_FACTOR, or reducing attenuation_db."
        )

    # Enforce anti-symmetry numerically
    window = 0.5 * (window - window[::-1])

    # If prototype was used, interpolate to original N
    if M != N:
        old_grid = np.linspace(-1, 1, M)
        new_grid = np.linspace(-1, 1, N)
        window = np.interp(new_grid, old_grid, window)

        # Re-enforce anti-symmetry after interpolation
        window = 0.5 * (window - window[::-1])

    window = _normalize_peak_abs(window, A)

    return window


def chebyshev_II_wca_pulses(t, pulse_start, T, A, attenuation_values):
    """
    Generates Chebyshev-II WCA/Remez pulse envelopes over:

        pulse_start <= t < pulse_start + T

    Outside this interval, the pulse is zero.
    """

    pulse_end = pulse_start + T

    mask = (t >= pulse_start) & (t < pulse_end)
    indices = np.where(mask)[0]
    N = len(indices)

    pulses = {}

    if N < 5:
        for attenuation_db in attenuation_values:
            pulses[attenuation_db] = np.zeros_like(t)
        return pulses, pulse_end, N

    for attenuation_db in attenuation_values:
        pulse = np.zeros_like(t)

        window = design_chebyshev_II_wca_window(
            N=N,
            attenuation_db=attenuation_db,
            A=A
        )

        pulse[indices] = window
        pulses[attenuation_db] = pulse

    return pulses, pulse_end, N


# ============================================================
# Main Loop
# ============================================================

plot_count = 0
data_count = 0

for T_ns in durations:

    T = T_ns * 1e-9

    for fs in fs_list:

        Ts = 1 / fs
        t = np.arange(t_start, t_end, Ts)

        cheb_pulses, pulse_end, N = chebyshev_II_wca_pulses(
            t=t,
            pulse_start=pulse_start,
            T=T,
            A=A,
            attenuation_values=attenuation_values
        )

        if N < 5:
            continue

        base_name = f"T{T_ns}ns_fs{fs/1e9:.1f}GSs"

        # ====================================================
        # 1. Chebyshev II WCA Baseband Pulse
        # ====================================================

        baseband_columns = [t * 1e9]
        baseband_header = "time_ns"

        plt.figure(figsize=(8, 4))

        for attenuation_db in attenuation_values:
            pulse = cheb_pulses[attenuation_db]

            baseband_columns.append(pulse)
            baseband_header += f",chebyshev_II_WCA_at{int(attenuation_db)}dB"

            plt.scatter(
                t * 1e9,
                pulse,
                s=8,
                label=f"Chebyshev II WCA, -{int(attenuation_db)} dB"
            )

        baseband_data = np.column_stack(baseband_columns)

        save_data(
            filename=f"01_baseband_chebyshev_II_WCA_{base_name}.csv",
            header=baseband_header,
            data_array=baseband_data
        )

        data_count += 1

        plt.title(
            f"Chebyshev II WCA Baseband Pulse | "
            f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | -60 dB"
        )

        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")

        plt.xlim(0, 2000)
        plt.ylim(-1.1, 1.1)

        plt.legend()
        plt.grid(True)

        finish_plot(
            filename=f"01_baseband_chebyshev_II_WCA_{base_name}.png"
        )

        plot_count += 1

        # ====================================================
        # 2. FFT of Chebyshev II WCA Baseband Pulse
        # ====================================================

        plt.figure(figsize=(8, 4))

        fft_columns = []
        fft_header = "frequency_GHz"

        first_freq = None

        for attenuation_db in attenuation_values:
            freq_base_pos, fft_db_pos = compute_fft_positive(
                cheb_pulses[attenuation_db],
                Ts
            )

            baseband_mask = (
                (freq_base_pos >= 0)
                &
                (freq_base_pos <= f_compare)
            )

            freq_base_plot = freq_base_pos[baseband_mask]
            fft_plot = fft_db_pos[baseband_mask]

            if first_freq is None:
                first_freq = freq_base_plot
                fft_columns.append(first_freq / 1e9)

            fft_columns.append(fft_plot)
            fft_header += f",chebyshev_II_WCA_at{int(attenuation_db)}dB_magnitude_dB"

            plt.plot(
                freq_base_plot / 1e9,
                fft_plot,
                linewidth=1.8,
                label=f"Chebyshev II WCA, -{int(attenuation_db)} dB"
            )

        fft_data = np.column_stack(fft_columns)

        save_data(
            filename=f"02_baseband_fft_chebyshev_II_WCA_{base_name}.csv",
            header=fft_header,
            data_array=fft_data
        )

        data_count += 1

        plt.axvline(
            0,
            color="k",
            linestyle=":",
            linewidth=1.5,
            label="0 Hz"
        )

        plt.title(
            f"Positive-Frequency FFT of Chebyshev II WCA Pulse | "
            f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | -60 dB"
        )

        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")

        plt.xlim(0, f_compare / 1e9)
        plt.ylim(-140, 5)

        plt.legend()
        plt.grid(True)

        finish_plot(
            filename=f"02_baseband_fft_chebyshev_II_WCA_{base_name}.png"
        )

        plot_count += 1

        # ====================================================
        # Carrier Frequency Variation
        # ====================================================

        for fc in fc_list:

            fc_name = f"fc{fc/1e6:.0f}MHz"
            carrier_iq = np.exp(1j * 2 * np.pi * fc * t)

            # ====================================================
            # 3. IQ-Modulated Chebyshev II WCA Pulse
            # ====================================================

            plt.figure(figsize=(10, 5))

            iq_columns = [t * 1e9]
            iq_header = "time_ns"

            iq_signals = {}

            for attenuation_db in attenuation_values:
                iq = cheb_pulses[attenuation_db] * carrier_iq
                iq_signals[attenuation_db] = iq

                I_component = np.real(iq)
                Q_component = np.imag(iq)

                iq_columns.append(I_component)
                iq_columns.append(Q_component)

                iq_header += (
                    f",chebyshev_II_WCA_at{int(attenuation_db)}dB_I,"
                    f"chebyshev_II_WCA_at{int(attenuation_db)}dB_Q"
                )

                plt.plot(t * 1e9, I_component, linewidth=0.8, alpha=0.55)
                plt.scatter(
                    t * 1e9,
                    I_component,
                    s=8,
                    marker="o",
                    label=f"Cheb II WCA -{int(attenuation_db)} dB I"
                )

                plt.plot(t * 1e9, Q_component, linewidth=0.8, alpha=0.55)
                plt.scatter(
                    t * 1e9,
                    Q_component,
                    s=8,
                    marker="x",
                    label=f"Cheb II WCA -{int(attenuation_db)} dB Q"
                )

            iq_time_data = np.column_stack(iq_columns)

            save_data(
                filename=f"03_iq_chebyshev_II_WCA_{base_name}_{fc_name}.csv",
                header=iq_header,
                data_array=iq_time_data
            )

            data_count += 1

            plt.title(
                f"Complex IQ-Modulated Chebyshev II WCA Pulse | "
                f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | "
                f"fc = {fc/1e6:.0f} MHz | -60 dB"
            )

            plt.xlabel("Time (ns)")
            plt.ylabel("Amplitude")

            plt.xlim(
                pulse_start * 1e9 - 50,
                pulse_end * 1e9 + 50
            )

            plt.ylim(-1.2, 1.2)

            plt.legend()
            plt.grid(True)

            finish_plot(
                filename=f"03_iq_chebyshev_II_WCA_{base_name}_{fc_name}.png"
            )

            plot_count += 1

            # ====================================================
            # 4. FFT of IQ-Modulated Chebyshev II WCA Pulse
            # ====================================================

            plt.figure(figsize=(8, 4))

            iq_fft_columns = []
            iq_fft_header = "frequency_GHz"
            first_freq_iq = None

            for attenuation_db in attenuation_values:
                freq_iq, iq_fft_db = compute_fft_complex(
                    iq_signals[attenuation_db],
                    Ts
                )

                iq_mask = (
                    (freq_iq >= -f_compare)
                    &
                    (freq_iq <= f_compare)
                )

                freq_iq_plot = freq_iq[iq_mask]
                iq_fft_plot = iq_fft_db[iq_mask]

                if first_freq_iq is None:
                    first_freq_iq = freq_iq_plot
                    iq_fft_columns.append(first_freq_iq / 1e9)

                iq_fft_columns.append(iq_fft_plot)
                iq_fft_header += f",chebyshev_II_WCA_at{int(attenuation_db)}dB_magnitude_dB"

                plt.plot(
                    freq_iq_plot / 1e9,
                    iq_fft_plot,
                    linewidth=1.8,
                    label=f"Cheb II WCA -{int(attenuation_db)} dB IQ FFT"
                )

            iq_fft_data = np.column_stack(iq_fft_columns)

            save_data(
                filename=f"04_iq_fft_chebyshev_II_WCA_{base_name}_{fc_name}.csv",
                header=iq_fft_header,
                data_array=iq_fft_data
            )

            data_count += 1

            plt.axvline(
                fc / 1e9,
                color="k",
                linestyle=":",
                linewidth=1.5,
                label=f"+fc = {fc/1e6:.0f} MHz"
            )

            plt.axvline(
                -fc / 1e9,
                color="gray",
                linestyle=":",
                linewidth=1.2,
                label=f"-fc = {fc/1e6:.0f} MHz"
            )

            plt.axvline(
                0,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label="0 Hz"
            )

            plt.title(
                f"Masked Full FFT of IQ Chebyshev II WCA Pulse | "
                f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | "
                f"fc = {fc/1e6:.0f} MHz | -60 dB"
            )

            plt.xlabel("Frequency (GHz)")
            plt.ylabel("Magnitude (dB)")

            plt.xlim(-f_compare / 1e9, f_compare / 1e9)
            plt.ylim(-140, 5)

            plt.legend()
            plt.grid(True)

            finish_plot(
                filename=f"04_iq_fft_chebyshev_II_WCA_{base_name}_{fc_name}.png"
            )

            plot_count += 1


print(f"Total plots generated: {plot_count}")
print(f"Total data files generated: {data_count}")
print(f"Plot folder: {plot_folder}")
print(f"Data folder: {data_folder}")

