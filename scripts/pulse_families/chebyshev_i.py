# Source notebook: Chebyshev_I.ipynb
# Source: https://colab.research.google.com/drive/12zkQry2_s07jnp9MD8ej7Vn-CNtXSOCZ
# Original calculation cell exported without changes.
# Colab-only ZIP and download cells are omitted.

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal.windows import chebwin

# ============================================================
# Plot and Data Control
# ============================================================

SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_DATA = True

main_folder = "chebyshev_I_study"
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

# Chebyshev I sidelobe attenuation values in dB
at_values = np.array([60, 80])

A = 1

t_start = 0
t_end = 2000e-9
pulse_start = 500e-9

f_compare = np.min(fs_list) / 2


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
# Chebyshev I Pulse Function
# ============================================================

def chebyshev_I_pulses(t, pulse_start, T, A, at_values):
    """
    Generates symmetric Dolph-Chebyshev / Chebyshev I pulse envelopes.

    The scipy chebwin window uses the attenuation parameter at in dB.
    Larger at means stronger sidelobe suppression but a wider main lobe.
    """

    pulse_end = pulse_start + T

    mask = (t >= pulse_start) & (t < pulse_end)
    indices = np.where(mask)[0]
    N = len(indices)

    pulses = {}

    if N < 2:
        for at_db in at_values:
            pulses[at_db] = np.zeros_like(t)
        return pulses, pulse_end, N

    for at_db in at_values:
        pulse = np.zeros_like(t)

        window = chebwin(M=N, at=at_db, sym=True)
        window = A * window / np.max(np.abs(window))

        pulse[indices] = window
        pulses[at_db] = pulse

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

        cheb_pulses, pulse_end, N = chebyshev_I_pulses(
            t=t,
            pulse_start=pulse_start,
            T=T,
            A=A,
            at_values=at_values
        )

        if N < 2:
            continue

        base_name = f"T{T_ns}ns_fs{fs/1e9:.1f}GSs"

        # ====================================================
        # 1. Chebyshev I Baseband Pulses
        # ====================================================

        baseband_columns = [t * 1e9]
        baseband_header = "time_ns"

        plt.figure(figsize=(8, 4))

        for at_db in at_values:
            pulse = cheb_pulses[at_db]

            baseband_columns.append(pulse)
            baseband_header += f",chebyshev_I_at{int(at_db)}dB"

            plt.scatter(
                t * 1e9,
                pulse,
                s=8,
                label=f"Chebyshev I, -{int(at_db)} dB"
            )

        baseband_data = np.column_stack(baseband_columns)

        save_data(
            filename=f"01_baseband_chebyshev_I_{base_name}.csv",
            header=baseband_header,
            data_array=baseband_data
        )

        data_count += 1

        plt.title(
            f"Chebyshev I Baseband Pulses | "
            f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s"
        )

        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")

        plt.xlim(0, 2000)
        plt.ylim(-0.1, 1.1)

        plt.legend()
        plt.grid(True)

        finish_plot(
            filename=f"01_baseband_chebyshev_I_{base_name}.png"
        )

        plot_count += 1

        # ====================================================
        # 2. FFT of Chebyshev I Baseband Pulses
        # ====================================================

        plt.figure(figsize=(8, 4))

        fft_columns = []
        fft_header = "frequency_GHz"

        first_freq = None

        for at_db in at_values:
            freq_base_pos, fft_db_pos = compute_fft_positive(
                cheb_pulses[at_db],
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
            fft_header += f",chebyshev_I_at{int(at_db)}dB_magnitude_dB"

            plt.plot(
                freq_base_plot / 1e9,
                fft_plot,
                linewidth=1.8,
                label=f"Chebyshev I, -{int(at_db)} dB"
            )

        fft_data = np.column_stack(fft_columns)

        save_data(
            filename=f"02_baseband_fft_chebyshev_I_{base_name}.csv",
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
            f"Positive-Frequency FFT of Chebyshev I Pulses | "
            f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s"
        )

        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")

        plt.xlim(0, f_compare / 1e9)
        plt.ylim(-140, 5)

        plt.legend()
        plt.grid(True)

        finish_plot(
            filename=f"02_baseband_fft_chebyshev_I_{base_name}.png"
        )

        plot_count += 1

        # ====================================================
        # Carrier Frequency Variation
        # ====================================================

        for fc in fc_list:

            fc_name = f"fc{fc/1e6:.0f}MHz"
            carrier_iq = np.exp(1j * 2 * np.pi * fc * t)

            # ====================================================
            # 3. IQ-Modulated Chebyshev I Pulses
            # ====================================================

            plt.figure(figsize=(10, 5))

            iq_columns = [t * 1e9]
            iq_header = "time_ns"

            iq_signals = {}

            for at_db in at_values:
                iq = cheb_pulses[at_db] * carrier_iq
                iq_signals[at_db] = iq

                I_component = np.real(iq)
                Q_component = np.imag(iq)

                iq_columns.append(I_component)
                iq_columns.append(Q_component)

                iq_header += (
                    f",chebyshev_I_at{int(at_db)}dB_I,"
                    f"chebyshev_I_at{int(at_db)}dB_Q"
                )

                plt.plot(t * 1e9, I_component, linewidth=0.8, alpha=0.55)
                plt.scatter(
                    t * 1e9,
                    I_component,
                    s=8,
                    marker="o",
                    label=f"Cheb I -{int(at_db)} dB I"
                )

                plt.plot(t * 1e9, Q_component, linewidth=0.8, alpha=0.55)
                plt.scatter(
                    t * 1e9,
                    Q_component,
                    s=8,
                    marker="x",
                    label=f"Cheb I -{int(at_db)} dB Q"
                )

            iq_time_data = np.column_stack(iq_columns)

            save_data(
                filename=f"03_iq_chebyshev_I_{base_name}_{fc_name}.csv",
                header=iq_header,
                data_array=iq_time_data
            )

            data_count += 1

            plt.title(
                f"Complex IQ-Modulated Chebyshev I Pulses | "
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
                filename=f"03_iq_chebyshev_I_{base_name}_{fc_name}.png"
            )

            plot_count += 1

            # ====================================================
            # 4. FFT of IQ-Modulated Chebyshev I Pulses
            # ====================================================

            plt.figure(figsize=(8, 4))

            iq_fft_columns = []
            iq_fft_header = "frequency_GHz"
            first_freq_iq = None

            for at_db in at_values:
                freq_iq, iq_fft_db = compute_fft_complex(
                    iq_signals[at_db],
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
                iq_fft_header += f",chebyshev_I_at{int(at_db)}dB_magnitude_dB"

                plt.plot(
                    freq_iq_plot / 1e9,
                    iq_fft_plot,
                    linewidth=1.8,
                    label=f"Cheb I -{int(at_db)} dB IQ FFT"
                )

            iq_fft_data = np.column_stack(iq_fft_columns)

            save_data(
                filename=f"04_iq_fft_chebyshev_I_{base_name}_{fc_name}.csv",
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
                f"Masked Full FFT of IQ Chebyshev I Pulses | "
                f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | "
                f"fc = {fc/1e6:.0f} MHz"
            )

            plt.xlabel("Frequency (GHz)")
            plt.ylabel("Magnitude (dB)")

            plt.xlim(-f_compare / 1e9, f_compare / 1e9)
            plt.ylim(-140, 5)

            plt.legend()
            plt.grid(True)

            finish_plot(
                filename=f"04_iq_fft_chebyshev_I_{base_name}_{fc_name}.png"
            )

            plot_count += 1


print(f"Total plots generated: {plot_count}")
print(f"Total data files generated: {data_count}")
print(f"Plot folder: {plot_folder}")
print(f"Data folder: {data_folder}")

