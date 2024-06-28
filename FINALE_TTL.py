import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def generate_waveform(pulse_width, start_frequency, end_frequency, baseline_time):
    num_pulses = 10  # Default number of pulses (you can adjust this based on your needs)
    frequency_increment = (end_frequency - start_frequency) / (num_pulses - 1)

    times = []
    values = []

    if baseline_time > 0:
        times.append(0)
        values.append(0)
        times.append(baseline_time)
        values.append(0)
        current_time = baseline_time
    else:
        current_time = 0

    for i in range(num_pulses):
        frequency = start_frequency + i * frequency_increment
        period = 1 / frequency
        high_time = pulse_width / 1000.0  # Convert ms to seconds
        low_time = period - high_time

        times.append(current_time)
        values.append(1)
        current_time += high_time

        times.append(current_time)
        values.append(0)
        current_time += low_time

    if baseline_time > 0:
        times.append(current_time)
        values.append(0)
        times.append(current_time + baseline_time)
        values.append(0)

    return times, values

def generate_shifted_waveform(times, values):
    high_to_low_time = None
    for i in range(1, len(values)):
        if values[i-1] == 1 and values[i] == 0:
            high_to_low_time = times[i]
            break
    
    low_to_high_time = None
    for i in range(len(values) - 1, 0, -1):
        if values[i-1] == 0 and values[i] == 1:
            low_to_high_time = times[i]
            break

    if high_to_low_time is None or low_to_high_time is None:
        raise ValueError("Not enough transitions to perform the shift.")
    
    shifted_times = [t for t in times]
    shifted_values = [0 if t < high_to_low_time else 1 - v for t, v in zip(times, values)]

    final_shifted_values = []
    for t, v in zip(shifted_times, shifted_values):
        if t >= low_to_high_time:
            final_shifted_values.append(0)
        else:
            final_shifted_values.append(v)

    return shifted_times, final_shifted_values

def generate_ttl_pulse_trains(frequency, total_time, num_samples, anticosine, baseline_time):
    time_step = total_time / num_samples

    time = np.arange(0, total_time, time_step)
    time_with_baseline = np.concatenate(([0, baseline_time], time + baseline_time, [total_time + baseline_time, total_time + baseline_time + baseline_time]))

    pulse_train1 = np.where(np.sin(2 * np.pi * frequency * time) >= 0, 1, 0)
    pulse_train1_with_baseline = np.concatenate(([0, 0], pulse_train1, [0, 0]))

    pulse_train2 = np.where(np.sin(2 * np.pi * frequency * time + anticosine) >= 0, 1, 0)
    pulse_train2_with_baseline = np.concatenate(([0, 0], pulse_train2, [0, 0]))

    return time_with_baseline, pulse_train1_with_baseline, pulse_train2_with_baseline

def compute_anticosine_correlation(correlation):
    correlation = np.clip(correlation, -1, 1)
    anticosine = np.arccos(correlation)
    return anticosine

def ttl_pulse_generator():
    st.title("TTL Pulse Generator")

    tab1, tab2, tab3 = st.tabs(["Pulse Width Modulated Waveform Generator", "Correlated Pulse Trains Generator", "Simple Waveform Generator"])

    baseline_time = st.number_input("Baseline Time (s):", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
    num_pairs = st.number_input("Number of Pairs of Waveforms:", min_value=1, max_value=10, step=1, value=1, format="%d")

    with tab1:
        st.header("Waveform Generator")

        pulse_width = st.number_input("Pulse Width (ms):", min_value=0.1, max_value=1000.0, step=0.1, value=100.0)
        start_frequency = st.number_input("Start Frequency (Hz):", min_value=0.1, max_value=1000.0, step=0.1, value=1.0)
        end_frequency = st.number_input("End Frequency (Hz):", min_value=0.1, max_value=1000.0, step=0.1, value=10.0)

        if st.button("Generate Waveforms"):
            st.write("Generating pulses...")

            num_waveforms = num_pairs * 2  # Each pair consists of two waveforms
            fig, axs = plt.subplots(num_waveforms, 1, figsize=(10, 4 * num_waveforms))

            if num_waveforms == 1:
                axs = [axs]  # Ensure axs is iterable

            for p in range(num_pairs):
                times, values = generate_waveform(pulse_width, start_frequency, end_frequency, baseline_time)
                for w in range(2):  # Generate two waveforms per pair
                    idx = p * 2 + w
                    if w == 0:
                        axs[idx].step(times, values, where='post')
                    else:
                        _, shifted_values = generate_shifted_waveform(times, values)
                        axs[idx].step(times, shifted_values, where='post')
                    axs[idx].set_xlabel("Time (s)")
                    axs[idx].set_ylabel("Pulse")
                    axs[idx].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            st.success("Pulses generated.")

    with tab2:
        st.header("Pulse Trains Generator")

        frequency = st.slider("Frequency (Hz)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
        total_time = st.slider("Total Time (s)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        num_samples = st.slider("Number of Samples", min_value=100, max_value=1000, value=500, step=50)
        correlation = st.slider("Correlation", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)

        if st.button("Generate Pulse Trains"):
            time, pulse_train1, pulse_train2 = generate_ttl_pulse_trains(frequency, total_time, num_samples, compute_anticosine_correlation(correlation), baseline_time)
            true_correlation = np.corrcoef(pulse_train1, pulse_train2)[0, 1]

            fig, ax = plt.subplots(num_pairs * 5, 1, figsize=(10, 12 * num_pairs))

            for i in range(num_pairs):
                idx1 = i * 5
                idx2 = i * 5 + 1
                idx3 = i * 5 + 2
                idx4 = i * 5 + 3
                idx5 = i * 5 + 4
                
                shifted_time1, shifted_pulse_train1 = generate_shifted_waveform(time, pulse_train1)
                shifted_time2, shifted_pulse_train2 = generate_shifted_waveform(time, pulse_train2)
                
                ax[idx1].plot(time, pulse_train1)
                ax[idx1].set_xlabel('Time (s)')
                ax[idx1].set_ylabel('Amplitude')
                ax[idx1].set_title(f'Pulse Train 1 - Pair {i+1}')

                ax[idx2].plot(shifted_time1, shifted_pulse_train1, color='green')
                ax[idx2].set_xlabel('Time (s)')
                ax[idx2].set_ylabel('Amplitude')
                ax[idx2].set_title(f'Shifted Pulse Train 1 - Pair {i+1}')

                ax[idx3].plot(time, pulse_train2, color='orange')
                ax[idx3].set_xlabel('Time (s)')
                ax[idx3].set_ylabel('Amplitude')
                ax[idx3].set_title(f'Pulse Train 2 - Pair {i+1}')

                ax[idx4].plot(shifted_time2, shifted_pulse_train2, color='red')
                ax[idx4].set_xlabel('Time (s)')
                ax[idx4].set_ylabel('Amplitude')
                ax[idx4].set_title(f'Shifted Pulse Train 2 - Pair {i+1}')

                ax[idx5].plot(time, pulse_train1)
                ax[idx5].plot(time, pulse_train2, color='orange')
                ax[idx5].set_xlabel('Time (s)')
                ax[idx5].set_ylabel('Amplitude')
                ax[idx5].set_title(f'Overlaid Pulses - Pair {i+1}')

                

                

                ax[idx1].text(0.02, 0.95, f'Intended Correlation: {correlation:.2f}', transform=ax[idx1].transAxes, verticalalignment='top')
                ax[idx1].text(0.02, 0.85, f'True Correlation: {true_correlation:.2f}', transform=ax[idx1].transAxes, verticalalignment='top')

            plt.tight_layout()
            st.pyplot(fig)

            st.success(f"Pulse trains generated with intended correlation {correlation:.2f} and true correlation {true_correlation:.2f}.")

    with tab3:
        st.header("Simple Waveform Generator")

        simple_start_frequency = st.number_input("Start Frequency (Hz):", min_value=0.1, max_value=1000.0, step=0.1, value=1.0, key="simple_start_frequency")
        simple_end_frequency = st.number_input("End Frequency (Hz):", min_value=0.1, max_value=1000.0, step=0.1, value=10.0, key="simple_end_frequency")
        simple_total_time = st.number_input("Total Time (s):", min_value=0.1, max_value=1000.0, step=0.1, value=10.0, key="simple_total_time")

        if st.button("Generate Simple Waveforms"):
            st.write("Generating simple waveforms...")

            avg_frequency = (simple_start_frequency + simple_end_frequency) / 2
            num_pulses = int(simple_total_time * avg_frequency)

            num_waveforms = num_pairs * 2  # Each pair consists of two waveforms
            fig, axs = plt.subplots(num_waveforms, 1, figsize=(10, 4 * num_waveforms))

            if num_waveforms == 1:
                axs = [axs]  # Ensure axs is iterable

            total_pulses = 0
            for p in range(num_pairs):
                times, values = generate_waveform(100.0, simple_start_frequency, simple_end_frequency, baseline_time)  # Using fixed pulse width
                for w in range(2):  # Generate two waveforms per pair
                    idx = p * 2 + w
                    if w == 0:
                        axs[idx].step(times, values, where='post')
                    else:
                        _, shifted_values = generate_shifted_waveform(times, values)
                        axs[idx].step(times, shifted_values, where='post')
                    total_pulses += len(times) // 2
                    axs[idx].set_xlabel("Time (s)")
                    axs[idx].set_ylabel("Pulse")
                    axs[idx].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            st.success(f"Simple waveforms generated. Total number of pulses: {total_pulses}")

if __name__ == "__main__":
    ttl_pulse_generator()
