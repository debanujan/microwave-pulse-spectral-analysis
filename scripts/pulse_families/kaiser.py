# Source notebook: Kaiser.ipynb
# Source: https://colab.research.google.com/drive/1Wv2p77eIAzAhfXNoqMre9hO0ptyvprfn
# Original calculation cell exported without changes.
# Colab-only ZIP and download cells are omitted.

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal.windows import kaiser

# ============================================================
# Plot and Data Control
# ============================================================

SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_DATA = True

main_folder = "kaiser_pulse_beta8_study"
plot_folder = os.path.join(main_folder, "plots")
data_folder = os.path.join(main_folder, "data")

if SAVE_PLOTS:
    os.makedirs(plot_folder, exist_ok=True)

if SAVE_DATA:
    os.makedirs(data_folder, exist_ok=True)


def finish_plot(filename):
    """
    Saves and/or shows each individual plot.
    """

    plt.tight_layout()

    if SAVE_PLOTS:
        filepath = os.path.join(plot_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def save_data(filename, header, data_array):
    """
    Saves numerical data used for each plot as a CSV file.
    """

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

# Pulse durations in ns
durations = np.array([50, 100, 250, 500, 750, 1000])

# Sampling rates in GS/s converted to Hz
fs_list = np.array([0.5, 1, 2, 4, 6, 10]) * 1e9

# Carrier frequencies in Hz
fc_list = np.array([50, 100, 150]) * 1e6

# Kaiser shape parameter
beta = 8

# Pulse amplitude
A = 1

# Global time window
t_start = 0
t_end = 2000e-9

# Pulse starts at 500 ns
pulse_start = 500e-9

# Common frequency comparison limit
# Lowest sampling rate = 0.5 GS/s
# Nyquist limit = 0.25 GHz
f_compare = np.min(fs_list) / 2


# ============================================================
# FFT Function: Positive Side for Real Baseband Signal
# ============================================================

def compute_fft_positive(signal, Ts):
    """
    Computes normalized FFT magnitude in dB
    and returns only the positive-frequency side.

    Suitable for real-valued baseband pulses.
    """

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


# ============================================================
# FFT Function: Full Spectrum for Complex IQ Signal
# ============================================================

def compute_fft_complex(signal, Ts):
    """
    Computes normalized FFT magnitude in dB
    and returns the full shifted spectrum from -fs/2 to +fs/2.

    Suitable for complex IQ signals.
    """

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
# Kaiser Pulse Function
# ============================================================

def kaiser_pulse(t, pulse_start, T, A, beta):
    """
    Generates a finite-duration Kaiser pulse over the interval

        pulse_start <= t < pulse_start + T

    Outside this interval, the pulse is zero.

    Kaiser window:
        w[n] = I0(beta sqrt(1 - x[n]^2)) / I0(beta)

    where x[n] varies from -1 to +1 over the pulse duration.

    beta controls the trade-off between main-lobe width and side-lobe level.
    Larger beta generally gives stronger side-lobe suppression but a wider
    main lobe.
    """

    pulse_end = pulse_start + T

    pulse = np.zeros_like(t)

    mask = (t >= pulse_start) & (t < pulse_end)
    indices = np.where(mask)[0]
    N = len(indices)

    if N < 2:
        return pulse, pulse_end, N

    window = kaiser(M=N, beta=beta, sym=True)

    # Normalize to peak amplitude A
    window = A * window / np.max(window)

    pulse[indices] = window

    return pulse, pulse_end, N


# ============================================================
# Main Loop
# ============================================================

plot_count = 0
data_count = 0

for T_ns in durations:

    # Convert duration from ns to seconds
    T = T_ns * 1e-9

    for fs in fs_list:

        # Sampling interval
        Ts = 1 / fs

        # Time axis
        t = np.arange(t_start, t_end, Ts)

        # Generate Kaiser baseband pulse
        kaiser_baseband, pulse_end, N = kaiser_pulse(
            t=t,
            pulse_start=pulse_start,
            T=T,
            A=A,
            beta=beta
        )

        if N < 2:
            continue

        base_name = f"T{T_ns}ns_fs{fs/1e9:.1f}GSs_beta{beta}"

        # ====================================================
        # 1. Kaiser Baseband Pulse
        # Total plots: 6 x 6 = 36
        # ====================================================

        baseband_data = np.column_stack(
            (
                t * 1e9,
                kaiser_baseband
            )
        )

        save_data(
            filename=f"01_baseband_kaiser_{base_name}.csv",
            header="time_ns,amplitude",
            data_array=baseband_data
        )

        data_count += 1

        plt.figure(figsize=(8, 4))

        plt.scatter(
            t * 1e9,
            kaiser_baseband,
            s=8,
            label=f"Kaiser beta={beta}"
        )

        plt.title(
            f"Kaiser Baseband Pulse | "
            f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | beta = {beta}"
        )

        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")

        plt.xlim(0, 2000)
        plt.ylim(-0.1, 1.1)

        plt.legend()
        plt.grid(True)

        finish_plot(
            filename=f"01_baseband_kaiser_{base_name}.png"
        )

        plot_count += 1

        # ====================================================
        # 2. FFT of Kaiser Baseband Pulse
        # Positive frequency side only
        # Masked from 0 to 0.25 GHz
        # Total plots: 6 x 6 = 36
        # ====================================================

        freq_base_pos, kaiser_fft_db_pos = compute_fft_positive(
            kaiser_baseband,
            Ts
        )

        baseband_mask = (
            (freq_base_pos >= 0)
            &
            (freq_base_pos <= f_compare)
        )

        freq_base_plot = freq_base_pos[baseband_mask]
        kaiser_fft_plot = kaiser_fft_db_pos[baseband_mask]

        baseband_fft_data = np.column_stack(
            (
                freq_base_plot / 1e9,
                kaiser_fft_plot
            )
        )

        save_data(
            filename=f"02_baseband_fft_kaiser_{base_name}.csv",
            header="frequency_GHz,magnitude_dB",
            data_array=baseband_fft_data
        )

        data_count += 1

        plt.figure(figsize=(8, 4))

        plt.plot(
            freq_base_plot / 1e9,
            kaiser_fft_plot,
            linewidth=1.8,
            label=f"Kaiser beta={beta} FFT"
        )

        plt.axvline(
            0,
            color="k",
            linestyle=":",
            linewidth=1.5,
            label="0 Hz"
        )

        plt.title(
            f"Positive-Frequency FFT of Kaiser Pulse | "
            f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | beta = {beta}"
        )

        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")

        plt.xlim(0, f_compare / 1e9)
        plt.ylim(-120, 5)

        plt.legend()
        plt.grid(True)

        finish_plot(
            filename=f"02_baseband_fft_kaiser_{base_name}.png"
        )

        plot_count += 1

        # ====================================================
        # Carrier Frequency Variation
        # ====================================================

        for fc in fc_list:

            fc_name = f"fc{fc/1e6:.0f}MHz"

            # ====================================================
            # Complex IQ Modulation
            # s_IQ(t) = w(t) exp(j 2 pi fc t)
            # ====================================================

            kaiser_iq = kaiser_baseband * np.exp(1j * 2 * np.pi * fc * t)

            kaiser_I = np.real(kaiser_iq)
            kaiser_Q = np.imag(kaiser_iq)

            # ====================================================
            # 3. Complex IQ-Modulated Kaiser Pulse
            # Scatter + light line plot
            # Total plots: 6 x 6 x 3 = 108
            # ====================================================

            iq_time_data = np.column_stack(
                (
                    t * 1e9,
                    kaiser_I,
                    kaiser_Q
                )
            )

            save_data(
                filename=f"03_iq_kaiser_{base_name}_{fc_name}.csv",
                header="time_ns,I_component,Q_component",
                data_array=iq_time_data
            )

            data_count += 1

            plt.figure(figsize=(10, 5))

            plt.plot(
                t * 1e9,
                kaiser_I,
                linewidth=0.8,
                alpha=0.6
            )

            plt.scatter(
                t * 1e9,
                kaiser_I,
                s=8,
                marker="o",
                label="I component"
            )

            plt.plot(
                t * 1e9,
                kaiser_Q,
                linewidth=0.8,
                alpha=0.6
            )

            plt.scatter(
                t * 1e9,
                kaiser_Q,
                s=8,
                marker="x",
                label="Q component"
            )

            plt.title(
                f"Complex IQ-Modulated Kaiser Pulse | "
                f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | "
                f"fc = {fc/1e6:.0f} MHz | beta = {beta}"
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
                filename=f"03_iq_kaiser_{base_name}_{fc_name}.png"
            )

            plot_count += 1

            # ====================================================
            # 4. FFT of Complex IQ-Modulated Kaiser Pulse
            # Full negative and positive frequency spectrum
            # Masked from -0.25 GHz to +0.25 GHz
            # Total plots: 6 x 6 x 3 = 108
            # ====================================================

            freq_iq, kaiser_iq_fft_db = compute_fft_complex(
                kaiser_iq,
                Ts
            )

            iq_mask = (
                (freq_iq >= -f_compare)
                &
                (freq_iq <= f_compare)
            )

            freq_iq_plot = freq_iq[iq_mask]
            kaiser_iq_fft_plot = kaiser_iq_fft_db[iq_mask]

            iq_fft_data = np.column_stack(
                (
                    freq_iq_plot / 1e9,
                    kaiser_iq_fft_plot
                )
            )

            save_data(
                filename=f"04_iq_fft_kaiser_{base_name}_{fc_name}.csv",
                header="frequency_GHz,magnitude_dB",
                data_array=iq_fft_data
            )

            data_count += 1

            plt.figure(figsize=(8, 4))

            plt.plot(
                freq_iq_plot / 1e9,
                kaiser_iq_fft_plot,
                linewidth=1.8,
                label=f"Kaiser beta={beta} IQ FFT"
            )

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
                label=f"-fc = {-fc/1e6:.0f} MHz"
            )

            plt.axvline(
                0,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label="0 Hz"
            )

            plt.title(
                f"Masked Full FFT of IQ Kaiser Pulse | "
                f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | "
                f"fc = {fc/1e6:.0f} MHz | beta = {beta}"
            )

            plt.xlabel("Frequency (GHz)")
            plt.ylabel("Magnitude (dB)")

            plt.xlim(-f_compare / 1e9, f_compare / 1e9)
            plt.ylim(-120, 5)

            plt.legend()
            plt.grid(True)

            finish_plot(
                filename=f"04_iq_fft_kaiser_{base_name}_{fc_name}.png"
            )

            plot_count += 1


print(f"Total plots generated: {plot_count}")
print(f"Total data files generated: {data_count}")
print(f"Plot folder: {plot_folder}")
print(f"Data folder: {data_folder}")

