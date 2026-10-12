# Source notebook: DRAG.ipynb
# Source: https://colab.research.google.com/drive/1UT6tXylIQi5m6u6FHNXYLBUrOyXSz8mO
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

main_folder = "drag_gaussian_beta1_pulse_study"
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

# Single DRAG strength used for this study.
# beta = 1.0 is the standard first-pass DRAG correction choice.
# beta = 0 would give ordinary Gaussian with Q = 0.
beta_values = np.array([1.0])

A = 1.0

# Transmon anharmonicity.
# Replace this with the actual measured anharmonicity when available.
# For a transmon, alpha is usually negative.
alpha_Hz = -250e6

# Same Gaussian protocol used earlier:
# large time window to observe the full rise/fall behaviour.
t_start = 0
t_end = 4000e-9
pulse_start = 1500e-9

# Duration convention:
# T_duration = 2 sigma, as used in the Gaussian study.
# Therefore sigma = T / 2.
SIGMA_DIVISOR = 2.0

# Lowest sampling rate = 0.5 GS/s, so common Nyquist comparison limit is 0.25 GHz.
f_compare = np.min(fs_list) / 2


# ============================================================
# FFT Functions
# ============================================================

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
# DRAG Gaussian Pulse Construction
# ============================================================

def normalize_peak(x, A):
    peak = np.max(np.abs(x))

    if peak == 0:
        return x

    return A * x / peak


def drag_gaussian_pulses(t, pulse_start, T, A, beta_values, alpha_Hz):
    """
    Generates Gaussian DRAG pulse envelopes.

    The ordinary Gaussian part is placed in the in-phase channel I(t).
    The derivative correction is placed in the quadrature channel Q(t).

    Complex baseband envelope:

        Omega(t) = I(t) + i Q(t)

    DRAG quadrature used here:

        Q(t) = - beta/(2*pi*alpha_Hz) * dI(t)/dt

    Notes:
    1. beta = 0 gives an ordinary Gaussian pulse.
    2. The sign of Q may depend on the laboratory IQ convention.
    3. alpha_Hz is taken negative for a transmon.
    4. The Gaussian is endpoint-corrected inside the pulse window so that
       I(t) starts and ends close to zero.
    """

    pulse_end = pulse_start + T

    mask = (t >= pulse_start) & (t < pulse_end)
    indices = np.where(mask)[0]
    N = len(indices)

    pulses = {}

    if N < 5:
        for beta in beta_values:
            pulses[beta] = {
                "I": np.zeros_like(t),
                "Q": np.zeros_like(t),
                "complex": np.zeros_like(t, dtype=complex)
            }

        return pulses, pulse_end, N

    sigma = T / SIGMA_DIVISOR
    t_center = pulse_start + T / 2

    tau = t[indices] - t_center

    # Raw Gaussian over the pulse interval
    gaussian_raw = np.exp(-tau ** 2 / (2 * sigma ** 2))

    # Endpoint correction.
    # At the two edges tau = +/- T/2.
    edge_value = np.exp(-((T / 2) ** 2) / (2 * sigma ** 2))

    gaussian_shifted = gaussian_raw - edge_value
    gaussian_shifted[gaussian_shifted < 0] = 0

    # Peak normalize I(t)
    I_local = normalize_peak(gaussian_shifted, A)

    # Derivative of the raw Gaussian.
    # The endpoint-shifting constant has zero derivative.
    d_gaussian_raw_dt = -(tau / sigma ** 2) * gaussian_raw

    # Apply the same peak normalization factor to the derivative.
    peak_before_normalization = np.max(np.abs(gaussian_shifted))

    if peak_before_normalization > 0:
        dI_local_dt = A * d_gaussian_raw_dt / peak_before_normalization
    else:
        dI_local_dt = d_gaussian_raw_dt

    I_full = np.zeros_like(t)
    dI_full_dt = np.zeros_like(t)

    I_full[indices] = I_local
    dI_full_dt[indices] = dI_local_dt

    for beta in beta_values:

        if np.abs(alpha_Hz) < 1e-30:
            Q_full = np.zeros_like(t)
        else:
            Q_full = -beta * dI_full_dt / (2 * np.pi * alpha_Hz)

        complex_envelope = I_full + 1j * Q_full

        pulses[beta] = {
            "I": I_full.copy(),
            "Q": Q_full.copy(),
            "complex": complex_envelope.copy()
        }

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

        drag_pulses, pulse_end, N = drag_gaussian_pulses(
            t=t,
            pulse_start=pulse_start,
            T=T,
            A=A,
            beta_values=beta_values,
            alpha_Hz=alpha_Hz
        )

        if N < 5:
            continue

        base_name = f"T{T_ns}ns_fs{fs/1e9:.1f}GSs"

        # ====================================================
        # 1. DRAG Gaussian Baseband Pulse
        # ====================================================

        baseband_columns = [t * 1e9]
        baseband_header = "time_ns"

        plt.figure(figsize=(10, 5))

        for beta in beta_values:

            I_component = drag_pulses[beta]["I"]
            Q_component = drag_pulses[beta]["Q"]

            beta_name = f"beta{beta:.1f}".replace(".", "p")

            baseband_columns.append(I_component)
            baseband_columns.append(Q_component)

            baseband_header += f",drag_{beta_name}_I,drag_{beta_name}_Q"

            plt.scatter(
                t * 1e9,
                I_component,
                s=8,
                marker="o",
                label=f"I, beta = {beta:.1f}"
            )

            plt.scatter(
                t * 1e9,
                Q_component,
                s=8,
                marker="x",
                label=f"Q, beta = {beta:.1f}"
            )

        baseband_data = np.column_stack(baseband_columns)

        save_data(
            filename=f"01_baseband_drag_{base_name}.csv",
            header=baseband_header,
            data_array=baseband_data
        )

        data_count += 1

        plt.title(
            f"Gaussian DRAG Baseband I/Q Pulse | "
            f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | "
            f"alpha = {alpha_Hz/1e6:.0f} MHz"
        )

        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")

        plt.xlim(
            pulse_start * 1e9 - 100,
            pulse_end * 1e9 + 100
        )

        plt.ylim(-1.2, 1.2)

        plt.legend(fontsize=8, ncol=2)
        plt.grid(True)

        finish_plot(
            filename=f"01_baseband_drag_{base_name}.png"
        )

        plot_count += 1

        # ====================================================
        # 2. FFT of DRAG Gaussian Baseband Complex Envelope
        # ====================================================
        #
        # Important:
        # DRAG baseband is complex because Omega(t) = I(t) + iQ(t).
        # Therefore, unlike a purely real pulse, the baseband FFT need
        # not be symmetric. We keep both positive and negative detuning.

        plt.figure(figsize=(9, 5))

        fft_columns = []
        fft_header = "frequency_GHz"
        first_freq = None

        for beta in beta_values:

            complex_envelope = drag_pulses[beta]["complex"]

            freq_base, fft_db = compute_fft_complex(
                complex_envelope,
                Ts
            )

            baseband_mask = (
                (freq_base >= -f_compare)
                &
                (freq_base <= f_compare)
            )

            freq_base_plot = freq_base[baseband_mask]
            fft_plot = fft_db[baseband_mask]

            if first_freq is None:
                first_freq = freq_base_plot
                fft_columns.append(first_freq / 1e9)

            beta_name = f"beta{beta:.1f}".replace(".", "p")

            fft_columns.append(fft_plot)
            fft_header += f",drag_{beta_name}_magnitude_dB"

            plt.plot(
                freq_base_plot / 1e9,
                fft_plot,
                linewidth=1.8,
                label=f"DRAG beta = {beta:.1f}"
            )

        fft_data = np.column_stack(fft_columns)

        save_data(
            filename=f"02_baseband_fft_drag_{base_name}.csv",
            header=fft_header,
            data_array=fft_data
        )

        data_count += 1

        plt.axvline(
            0,
            color="k",
            linestyle=":",
            linewidth=1.5,
            label="0 detuning"
        )

        plt.axvline(
            alpha_Hz / 1e9,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"alpha = {alpha_Hz/1e6:.0f} MHz"
        )

        plt.title(
            f"Full FFT of DRAG Baseband Envelope | "
            f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s"
        )

        plt.xlabel("Frequency / detuning (GHz)")
        plt.ylabel("Magnitude (dB)")

        plt.xlim(-f_compare / 1e9, f_compare / 1e9)
        plt.ylim(-140, 5)

        plt.legend(fontsize=8)
        plt.grid(True)

        finish_plot(
            filename=f"02_baseband_fft_drag_{base_name}.png"
        )

        plot_count += 1

        # ====================================================
        # Carrier Frequency Variation
        # ====================================================

        for fc in fc_list:

            fc_name = f"fc{fc/1e6:.0f}MHz"
            carrier_iq = np.exp(1j * 2 * np.pi * fc * t)

            # ====================================================
            # 3. IQ-Modulated DRAG Gaussian Pulse
            # ====================================================

            plt.figure(figsize=(10, 5))

            iq_columns = [t * 1e9]
            iq_header = "time_ns"

            iq_signals = {}

            for beta in beta_values:

                complex_envelope = drag_pulses[beta]["complex"]

                iq = complex_envelope * carrier_iq
                iq_signals[beta] = iq

                I_component = np.real(iq)
                Q_component = np.imag(iq)

                beta_name = f"beta{beta:.1f}".replace(".", "p")

                iq_columns.append(I_component)
                iq_columns.append(Q_component)

                iq_header += f",drag_{beta_name}_I,drag_{beta_name}_Q"

                plt.plot(
                    t * 1e9,
                    I_component,
                    linewidth=0.8,
                    alpha=0.55
                )

                plt.scatter(
                    t * 1e9,
                    I_component,
                    s=8,
                    marker="o",
                    label=f"DRAG beta {beta:.1f} I"
                )

                plt.plot(
                    t * 1e9,
                    Q_component,
                    linewidth=0.8,
                    alpha=0.55
                )

                plt.scatter(
                    t * 1e9,
                    Q_component,
                    s=8,
                    marker="x",
                    label=f"DRAG beta {beta:.1f} Q"
                )

            iq_time_data = np.column_stack(iq_columns)

            save_data(
                filename=f"03_iq_drag_{base_name}_{fc_name}.csv",
                header=iq_header,
                data_array=iq_time_data
            )

            data_count += 1

            plt.title(
                f"Complex IQ-Modulated Gaussian DRAG Pulse | "
                f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | "
                f"fc = {fc/1e6:.0f} MHz"
            )

            plt.xlabel("Time (ns)")
            plt.ylabel("Amplitude")

            plt.xlim(
                pulse_start * 1e9 - 100,
                pulse_end * 1e9 + 100
            )

            plt.ylim(-1.2, 1.2)

            plt.legend(fontsize=7, ncol=2)
            plt.grid(True)

            finish_plot(
                filename=f"03_iq_drag_{base_name}_{fc_name}.png"
            )

            plot_count += 1

            # ====================================================
            # 4. FFT of IQ-Modulated DRAG Gaussian Pulse
            # ====================================================

            plt.figure(figsize=(9, 5))

            iq_fft_columns = []
            iq_fft_header = "frequency_GHz"
            first_freq_iq = None

            for beta in beta_values:

                freq_iq, iq_fft_db = compute_fft_complex(
                    iq_signals[beta],
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

                beta_name = f"beta{beta:.1f}".replace(".", "p")

                iq_fft_columns.append(iq_fft_plot)
                iq_fft_header += f",drag_{beta_name}_magnitude_dB"

                plt.plot(
                    freq_iq_plot / 1e9,
                    iq_fft_plot,
                    linewidth=1.8,
                    label=f"DRAG beta = {beta:.1f}"
                )

            iq_fft_data = np.column_stack(iq_fft_columns)

            save_data(
                filename=f"04_iq_fft_drag_{base_name}_{fc_name}.csv",
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
                (fc + alpha_Hz) / 1e9,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label="fc + alpha"
            )

            plt.axvline(
                0,
                color="orange",
                linestyle="--",
                linewidth=1.0,
                label="0 Hz"
            )

            plt.title(
                f"Masked Full FFT of IQ Gaussian DRAG Pulse | "
                f"T = {T_ns} ns | fs = {fs/1e9:.1f} GS/s | "
                f"fc = {fc/1e6:.0f} MHz"
            )

            plt.xlabel("Frequency (GHz)")
            plt.ylabel("Magnitude (dB)")

            plt.xlim(-f_compare / 1e9, f_compare / 1e9)
            plt.ylim(-140, 5)

            plt.legend(fontsize=8)
            plt.grid(True)

            finish_plot(
                filename=f"04_iq_fft_drag_{base_name}_{fc_name}.png"
            )

            plot_count += 1


print(f"Total plots generated: {plot_count}")
print(f"Total data files generated: {data_count}")
print(f"Plot folder: {plot_folder}")
print(f"Data folder: {data_folder}")

