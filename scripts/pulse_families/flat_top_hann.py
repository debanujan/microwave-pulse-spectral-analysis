# Source notebook: Flat-top-cosine.ipynb
# Source: https://colab.research.google.com/drive/1lWOJO38H8qh9ycYDSnVSgGZdCODKQt2I
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

main_folder = "flat_top_hann_f05_2000ns_study"
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

# Total pulse durations in ns
durations = np.array([50, 100, 250, 500, 750, 1000])

# Sampling rates in GS/s converted to Hz
fs_list = np.array([0.5, 1, 2, 4, 6, 10]) * 1e9

# Carrier frequencies in Hz
fc_list = np.array([50, 100, 150]) * 1e6

# Flat fraction
# f_flat = T_flat / T_total
flat_fraction = 0.5

# Pulse amplitude
A = 1

# Global time window: 0 ns to 2000 ns
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
# Flat-Top Hann Rise/Fall Pulse Function
# ============================================================

def flat_top_hann_pulse(t, pulse_start, T_total, A, flat_fraction):
    """
    Generates a flat-top pulse with Hann-shaped rise and fall.

    Total duration:
        T_total = T_rise + T_flat + T_fall

    Flat fraction:
        flat_fraction = T_flat / T_total

    Therefore:
        T_flat = flat_fraction * T_total
        T_rise = T_fall = (T_total - T_flat) / 2

    Hann-shaped rise:
        w_rise(t) = A/2 * [1 - cos(pi * tau / T_rise)]

    where:
        tau = t - pulse_start
        0 <= tau < T_rise

    Hann-shaped fall:
        w_fall(t) = A/2 * [1 + cos(pi * tau / T_fall)]

    where:
        tau = t - flat_end
        0 <= tau < T_fall

    No sigma, beta, or extra shape parameter is used.
    Once flat_fraction is fixed, the pulse shape is fully determined.
    """

    pulse_end = pulse_start + T_total

    T_flat = flat_fraction * T_total
    T_edge_total = T_total - T_flat
    T_rise = T_edge_total / 2
    T_fall = T_edge_total / 2

    flat_start = pulse_start + T_rise
    flat_end = flat_start + T_flat

    pulse = np.zeros_like(t)

    # Rising Hann edge
    rise_mask = (t >= pulse_start) & (t < flat_start)
    tau_rise = t[rise_mask] - pulse_start
    pulse[rise_mask] = (A / 2) * (1 - np.cos(np.pi * tau_rise / T_rise))

    # Flat-top region
    flat_mask = (t >= flat_start) & (t < flat_end)
    pulse[flat_mask] = A

    # Falling Hann edge
    fall_mask = (t >= flat_end) & (t < pulse_end)
    tau_fall = t[fall_mask] - flat_end
    pulse[fall_mask] = (A / 2) * (1 + np.cos(np.pi * tau_fall / T_fall))

    return pulse, pulse_end, flat_start, flat_end, T_rise, T_flat, T_fall


# ============================================================
# Main Loop
# ============================================================

plot_count = 0
data_count = 0

for T_ns in durations:

    # Convert total duration from ns to seconds
    T_total = T_ns * 1e-9

    for fs in fs_list:

        # Sampling interval
        Ts = 1 / fs

        # Time axis
        t = np.arange(t_start, t_end, Ts)

        # Generate flat-top Hann baseband pulse
        fth_pulse, pulse_end, flat_start, flat_end, T_rise, T_flat, T_fall = flat_top_hann_pulse(
            t=t,
            pulse_start=pulse_start,
            T_total=T_total,
            A=A,
            flat_fraction=flat_fraction
        )

        base_name = f"T{T_ns}ns_fflat{flat_fraction:.2f}_fs{fs/1e9:.1f}GSs"

        # ====================================================
        # 1. Flat-Top Hann Baseband Pulse
        # Total plots: 6 x 6 = 36
        # ====================================================

        baseband_data = np.column_stack(
            (
                t * 1e9,
                fth_pulse
            )
        )

        save_data(
            filename=f"01_baseband_flat_top_hann_{base_name}.csv",
            header="time_ns,amplitude",
            data_array=baseband_data
        )

        data_count += 1

        plt.figure(figsize=(8, 4))

        plt.scatter(
            t * 1e9,
            fth_pulse,
            s=8,
            label="Flat-top Hann"
        )

        plt.axvline(
            pulse_start * 1e9,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label="pulse start"
        )

        plt.axvline(
            flat_start * 1e9,
            color="k",
            linestyle=":",
            linewidth=1.2,
            label="flat start"
        )

        plt.axvline(
            flat_end * 1e9,
            color="k",
            linestyle=":",
            linewidth=1.2,
            label="flat end"
        )

        plt.axvline(
            pulse_end * 1e9,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label="pulse end"
        )

        plt.title(
            f"Flat-Top Hann Baseband Pulse | "
            f"T = {T_ns} ns | f_flat = {flat_fraction:.2f} | fs = {fs/1e9:.1f} GS/s"
        )

        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")

        plt.xlim(0, 2000)
        plt.ylim(-0.1, 1.1)

        plt.legend()
        plt.grid(True)

        finish_plot(
            filename=f"01_baseband_flat_top_hann_{base_name}.png"
        )

        plot_count += 1

        # ====================================================
        # 2. FFT of Flat-Top Hann Baseband Pulse
        # Positive frequency side only
        # Masked from 0 to 0.25 GHz
        # Total plots: 6 x 6 = 36
        # ====================================================

        freq_base_pos, fth_fft_db_pos = compute_fft_positive(
            fth_pulse,
            Ts
        )

        baseband_mask = (
            (freq_base_pos >= 0)
            &
            (freq_base_pos <= f_compare)
        )

        freq_base_plot = freq_base_pos[baseband_mask]
        fth_fft_plot = fth_fft_db_pos[baseband_mask]

        baseband_fft_data = np.column_stack(
            (
                freq_base_plot / 1e9,
                fth_fft_plot
            )
        )

        save_data(
            filename=f"02_baseband_fft_flat_top_hann_{base_name}.csv",
            header="frequency_GHz,magnitude_dB",
            data_array=baseband_fft_data
        )

        data_count += 1

        plt.figure(figsize=(8, 4))

        plt.plot(
            freq_base_plot / 1e9,
            fth_fft_plot,
            linewidth=1.8,
            label="Flat-top Hann FFT"
        )

        plt.axvline(
            0,
            color="k",
            linestyle=":",
            linewidth=1.5,
            label="0 Hz"
        )

        plt.title(
            f"Positive-Frequency FFT of Flat-Top Hann Pulse | "
            f"T = {T_ns} ns | f_flat = {flat_fraction:.2f} | fs = {fs/1e9:.1f} GS/s"
        )

        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")

        plt.xlim(0, f_compare / 1e9)
        plt.ylim(-120, 5)

        plt.legend()
        plt.grid(True)

        finish_plot(
            filename=f"02_baseband_fft_flat_top_hann_{base_name}.png"
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

            fth_iq = fth_pulse * np.exp(1j * 2 * np.pi * fc * t)

            fth_I = np.real(fth_iq)
            fth_Q = np.imag(fth_iq)

            # ====================================================
            # 3. Complex IQ-Modulated Flat-Top Hann Pulse
            # Scatter + light line plot
            # Total plots: 6 x 6 x 3 = 108
            # ====================================================

            iq_time_data = np.column_stack(
                (
                    t * 1e9,
                    fth_I,
                    fth_Q
                )
            )

            save_data(
                filename=f"03_iq_flat_top_hann_{base_name}_{fc_name}.csv",
                header="time_ns,I_component,Q_component",
                data_array=iq_time_data
            )

            data_count += 1

            plt.figure(figsize=(10, 5))

            plt.plot(
                t * 1e9,
                fth_I,
                linewidth=0.8,
                alpha=0.6
            )

            plt.scatter(
                t * 1e9,
                fth_I,
                s=8,
                marker="o",
                label="I component"
            )

            plt.plot(
                t * 1e9,
                fth_Q,
                linewidth=0.8,
                alpha=0.6
            )

            plt.scatter(
                t * 1e9,
                fth_Q,
                s=8,
                marker="x",
                label="Q component"
            )

            plt.axvline(
                pulse_start * 1e9,
                color="gray",
                linestyle=":",
                linewidth=1.0,
                label="pulse start"
            )

            plt.axvline(
                flat_start * 1e9,
                color="k",
                linestyle=":",
                linewidth=1.0,
                label="flat start"
            )

            plt.axvline(
                flat_end * 1e9,
                color="k",
                linestyle=":",
                linewidth=1.0,
                label="flat end"
            )

            plt.axvline(
                pulse_end * 1e9,
                color="gray",
                linestyle=":",
                linewidth=1.0,
                label="pulse end"
            )

            plt.title(
                f"Complex IQ-Modulated Flat-Top Hann Pulse | "
                f"T = {T_ns} ns | f_flat = {flat_fraction:.2f} | fs = {fs/1e9:.1f} GS/s | "
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
                filename=f"03_iq_flat_top_hann_{base_name}_{fc_name}.png"
            )

            plot_count += 1

            # ====================================================
            # 4. FFT of Complex IQ-Modulated Flat-Top Hann Pulse
            # Full negative and positive frequency spectrum
            # Masked from -0.25 GHz to +0.25 GHz
            # Total plots: 6 x 6 x 3 = 108
            # ====================================================

            freq_iq, fth_iq_fft_db = compute_fft_complex(
                fth_iq,
                Ts
            )

            iq_mask = (
                (freq_iq >= -f_compare)
                &
                (freq_iq <= f_compare)
            )

            freq_iq_plot = freq_iq[iq_mask]
            fth_iq_fft_plot = fth_iq_fft_db[iq_mask]

            iq_fft_data = np.column_stack(
                (
                    freq_iq_plot / 1e9,
                    fth_iq_fft_plot
                )
            )

            save_data(
                filename=f"04_iq_fft_flat_top_hann_{base_name}_{fc_name}.csv",
                header="frequency_GHz,magnitude_dB",
                data_array=iq_fft_data
            )

            data_count += 1

            plt.figure(figsize=(8, 4))

            plt.plot(
                freq_iq_plot / 1e9,
                fth_iq_fft_plot,
                linewidth=1.8,
                label="Flat-top Hann IQ FFT"
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
                f"Masked Full FFT of IQ Flat-Top Hann Pulse | "
                f"T = {T_ns} ns | f_flat = {flat_fraction:.2f} | fs = {fs/1e9:.1f} GS/s | "
                f"fc = {fc/1e6:.0f} MHz"
            )

            plt.xlabel("Frequency (GHz)")
            plt.ylabel("Magnitude (dB)")

            plt.xlim(-f_compare / 1e9, f_compare / 1e9)
            plt.ylim(-120, 5)

            plt.legend()
            plt.grid(True)

            finish_plot(
                filename=f"04_iq_fft_flat_top_hann_{base_name}_{fc_name}.png"
            )

            plot_count += 1


print(f"Total plots generated: {plot_count}")
print(f"Total data files generated: {data_count}")
print(f"Plot folder: {plot_folder}")
print(f"Data folder: {data_folder}")

