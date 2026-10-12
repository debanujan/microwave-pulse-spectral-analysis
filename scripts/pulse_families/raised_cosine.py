# Source notebook: Raised Cosine Pulse.ipynb
# Source: https://colab.research.google.com/drive/1tVg5Kj73dQ9wEVl7J6xM20ZXLuvk8iGz
# Original calculation cell exported without changes.
# Colab-only ZIP and download cells are omitted.

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Plot and Data Control
# ============================================================

SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_DATA = True

main_folder = "raised_cosine_window_study"
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
# Raised-Cosine Window Pulse Function
# ============================================================

def raised_cosine_window_pulses(t, pulse_start, T, A):
    """
    Generates Hann, Hamming, and Blackman pulses over the interval

        pulse_start <= t < pulse_start + T

    Outside this interval, each pulse is zero.

    These are finite-duration raised-cosine / generalized cosine windows.

    Hann:
        w[n] = 0.5 - 0.5 cos(2 pi n / (N - 1))

    Hamming:
        w[n] = 0.54 - 0.46 cos(2 pi n / (N - 1))

    Blackman:
        w[n] = 0.42 - 0.5 cos(2 pi n / (N - 1))
               + 0.08 cos(4 pi n / (N - 1))
    """

    pulse_end = pulse_start + T

    hann_pulse = np.zeros_like(t)
    hamming_pulse = np.zeros_like(t)
    blackman_pulse = np.zeros_like(t)

    mask = (t >= pulse_start) & (t < pulse_end)
    indices = np.where(mask)[0]
    N = len(indices)

    if N < 2:
        return hann_pulse, hamming_pulse, blackman_pulse, pulse_end, N

    n = np.arange(N)

    hann_window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))

    hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

    blackman_window = (
        0.42
        - 0.5 * np.cos(2 * np.pi * n / (N - 1))
        + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    )

    # Normalize each pulse to peak amplitude A
    hann_window = A * hann_window / np.max(hann_window)
    hamming_window = A * hamming_window / np.max(hamming_window)
    blackman_window = A * blackman_window / np.max(blackman_window)

    hann_pulse[indices] = hann_window
    hamming_pulse[indices] = hamming_window
    blackman_pulse[indices] = blackman_window

    return hann_pulse, hamming_pulse, blackman_pulse, pulse_end, N


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

        # Generate raised-cosine window pulses
        hann_pulse, hamming_pulse, blackman_pulse, pulse_end, N = raised_cosine_window_pulses(
            t=t,
            pulse_start=pulse_start,
            T=T,
            A=A
        )

        if N < 2:
            continue

        base_name = f"T{T_ns}ns_fs{fs/1e9:.1f}GSs"

        # ====================================================
        # 1. Baseband Raised-Cosine Window Pulses
        # Total plots: 6 x 6 = 36
        # ====================================================

        baseband_data = np.column_stack(
            (
                t * 1e9,
                hann_pulse,
                hamming_pulse,
                blackman_pulse
            )
        )

        save_data(
            filename=f"01_baseband_raised_cosines_{base_name}.csv",
            header="time_ns,hann_amplitude,hamming_amplitude,blackman_amplitude",
            data_array=baseband_data
        )

        data_count += 1

        plt.figure(figsize=(8, 4))

        plt.scatter(
            t * 1e9,
            hann_pulse,
            s=8,
            label="Hann"
        )

        plt.scatter(
            t * 1e9,
            hamming_pulse,
            s=8,
            label="Hamming"
        )

        plt.scatter(
            t * 1e9,
            blackman_pulse,
            s=8,
            label="Blackman"
        )

        plt.title(
            f"Raised-Cosine Baseband Pulses | "
            f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s"
        )

        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")

        plt.xlim(0, 2000)
        plt.ylim(-0.1, 1.1)

        plt.legend()
        plt.grid(True)

        finish_plot(
            filename=f"01_baseband_raised_cosines_{base_name}.png"
        )

        plot_count += 1

        # ====================================================
        # 2. FFT of Baseband Raised-Cosine Window Pulses
        # Positive frequency side only
        # Masked from 0 to 0.25 GHz
        # Total plots: 6 x 6 = 36
        # ====================================================

        freq_base_pos, hann_fft_db_pos = compute_fft_positive(hann_pulse, Ts)
        _, hamming_fft_db_pos = compute_fft_positive(hamming_pulse, Ts)
        _, blackman_fft_db_pos = compute_fft_positive(blackman_pulse, Ts)

        baseband_mask = (
            (freq_base_pos >= 0)
            &
            (freq_base_pos <= f_compare)
        )

        freq_base_plot = freq_base_pos[baseband_mask]
        hann_fft_plot = hann_fft_db_pos[baseband_mask]
        hamming_fft_plot = hamming_fft_db_pos[baseband_mask]
        blackman_fft_plot = blackman_fft_db_pos[baseband_mask]

        baseband_fft_data = np.column_stack(
            (
                freq_base_plot / 1e9,
                hann_fft_plot,
                hamming_fft_plot,
                blackman_fft_plot
            )
        )

        save_data(
            filename=f"02_baseband_fft_raised_cosines_{base_name}.csv",
            header="frequency_GHz,hann_magnitude_dB,hamming_magnitude_dB,blackman_magnitude_dB",
            data_array=baseband_fft_data
        )

        data_count += 1

        plt.figure(figsize=(8, 4))

        plt.plot(
            freq_base_plot / 1e9,
            hann_fft_plot,
            linewidth=1.8,
            label="Hann FFT"
        )

        plt.plot(
            freq_base_plot / 1e9,
            hamming_fft_plot,
            linewidth=1.8,
            label="Hamming FFT"
        )

        plt.plot(
            freq_base_plot / 1e9,
            blackman_fft_plot,
            linewidth=1.8,
            label="Blackman FFT"
        )

        plt.axvline(
            0,
            color="k",
            linestyle=":",
            linewidth=1.5,
            label="0 Hz"
        )

        plt.title(
            f"Positive-Frequency FFT of Raised-Cosine Pulses | "
            f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s"
        )

        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")

        plt.xlim(0, f_compare / 1e9)
        plt.ylim(-120, 5)

        plt.legend()
        plt.grid(True)

        finish_plot(
            filename=f"02_baseband_fft_raised_cosines_{base_name}.png"
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

            carrier_iq = np.exp(1j * 2 * np.pi * fc * t)

            hann_iq = hann_pulse * carrier_iq
            hamming_iq = hamming_pulse * carrier_iq
            blackman_iq = blackman_pulse * carrier_iq

            hann_I = np.real(hann_iq)
            hann_Q = np.imag(hann_iq)

            hamming_I = np.real(hamming_iq)
            hamming_Q = np.imag(hamming_iq)

            blackman_I = np.real(blackman_iq)
            blackman_Q = np.imag(blackman_iq)

            # ====================================================
            # 3. Complex IQ-Modulated Raised-Cosine Pulses
            # Scatter + light line plot
            # Total plots: 6 x 6 x 3 = 108
            # ====================================================

            iq_time_data = np.column_stack(
                (
                    t * 1e9,
                    hann_I,
                    hann_Q,
                    hamming_I,
                    hamming_Q,
                    blackman_I,
                    blackman_Q
                )
            )

            save_data(
                filename=f"03_iq_raised_cosines_{base_name}_{fc_name}.csv",
                header=(
                    "time_ns,"
                    "hann_I,hann_Q,"
                    "hamming_I,hamming_Q,"
                    "blackman_I,blackman_Q"
                ),
                data_array=iq_time_data
            )

            data_count += 1

            plt.figure(figsize=(10, 5))

            # Hann
            plt.plot(t * 1e9, hann_I, linewidth=0.8, alpha=0.55)
            plt.scatter(t * 1e9, hann_I, s=8, marker="o", label="Hann I")
            plt.plot(t * 1e9, hann_Q, linewidth=0.8, alpha=0.55)
            plt.scatter(t * 1e9, hann_Q, s=8, marker="x", label="Hann Q")

            # Hamming
            plt.plot(t * 1e9, hamming_I, linewidth=0.8, alpha=0.55)
            plt.scatter(t * 1e9, hamming_I, s=8, marker="o", label="Hamming I")
            plt.plot(t * 1e9, hamming_Q, linewidth=0.8, alpha=0.55)
            plt.scatter(t * 1e9, hamming_Q, s=8, marker="x", label="Hamming Q")

            # Blackman
            plt.plot(t * 1e9, blackman_I, linewidth=0.8, alpha=0.55)
            plt.scatter(t * 1e9, blackman_I, s=8, marker="o", label="Blackman I")
            plt.plot(t * 1e9, blackman_Q, linewidth=0.8, alpha=0.55)
            plt.scatter(t * 1e9, blackman_Q, s=8, marker="x", label="Blackman Q")

            plt.title(
                f"Complex IQ-Modulated Raised-Cosine Pulses | "
                f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | "
                f"fc = {fc/1e6:.0f} MHz"
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
                filename=f"03_iq_raised_cosines_{base_name}_{fc_name}.png"
            )

            plot_count += 1

            # ====================================================
            # 4. FFT of Complex IQ-Modulated Raised-Cosine Pulses
            # Full negative and positive frequency spectrum
            # Masked from -0.25 GHz to +0.25 GHz
            # Total plots: 6 x 6 x 3 = 108
            # ====================================================

            freq_iq, hann_iq_fft_db = compute_fft_complex(hann_iq, Ts)
            _, hamming_iq_fft_db = compute_fft_complex(hamming_iq, Ts)
            _, blackman_iq_fft_db = compute_fft_complex(blackman_iq, Ts)

            iq_mask = (
                (freq_iq >= -f_compare)
                &
                (freq_iq <= f_compare)
            )

            freq_iq_plot = freq_iq[iq_mask]
            hann_iq_fft_plot = hann_iq_fft_db[iq_mask]
            hamming_iq_fft_plot = hamming_iq_fft_db[iq_mask]
            blackman_iq_fft_plot = blackman_iq_fft_db[iq_mask]

            iq_fft_data = np.column_stack(
                (
                    freq_iq_plot / 1e9,
                    hann_iq_fft_plot,
                    hamming_iq_fft_plot,
                    blackman_iq_fft_plot
                )
            )

            save_data(
                filename=f"04_iq_fft_raised_cosines_{base_name}_{fc_name}.csv",
                header=(
                    "frequency_GHz,"
                    "hann_magnitude_dB,"
                    "hamming_magnitude_dB,"
                    "blackman_magnitude_dB"
                ),
                data_array=iq_fft_data
            )

            data_count += 1

            plt.figure(figsize=(8, 4))

            plt.plot(
                freq_iq_plot / 1e9,
                hann_iq_fft_plot,
                linewidth=1.8,
                label="Hann IQ FFT"
            )

            plt.plot(
                freq_iq_plot / 1e9,
                hamming_iq_fft_plot,
                linewidth=1.8,
                label="Hamming IQ FFT"
            )

            plt.plot(
                freq_iq_plot / 1e9,
                blackman_iq_fft_plot,
                linewidth=1.8,
                label="Blackman IQ FFT"
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
                f"Masked Full FFT of IQ Raised-Cosine Pulses | "
                f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | "
                f"fc = {fc/1e6:.0f} MHz"
            )

            plt.xlabel("Frequency (GHz)")
            plt.ylabel("Magnitude (dB)")

            plt.xlim(-f_compare / 1e9, f_compare / 1e9)
            plt.ylim(-120, 5)

            plt.legend()
            plt.grid(True)

            finish_plot(
                filename=f"04_iq_fft_raised_cosines_{base_name}_{fc_name}.png"
            )

            plot_count += 1


print(f"Total plots generated: {plot_count}")
print(f"Total data files generated: {data_count}")
print(f"Plot folder: {plot_folder}")
print(f"Data folder: {data_folder}")

