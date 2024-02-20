from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal
from scipy.signal import *


# FUNCTIONS

def create_label_indicator(label_text, alignment):
    label = QLabel(label_text)

    indicator = QLabel('1')
    indicator.setAlignment(Qt.AlignLeft | (Qt.AlignTop if alignment == 1 else Qt.AlignBottom))

    label.setAlignment(Qt.AlignLeft | (Qt.AlignBottom if alignment == 1 else Qt.AlignVCenter))

    return label, indicator


def create_sliders(sliders_number, labels_list, frame, alignment):
    clear(frame)
    sliders_list = []
    indicators = []

    layout = frame.layout()

    for i in range(sliders_number):
        slider = QSlider()
        slider.setRange(0, 10)
        slider.setValue(1)
        slider.setOrientation(alignment)

        label, indicator = create_label_indicator(labels_list[i], alignment)

        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(indicator)

        sliders_list.append(slider)
        indicators.append(indicator)

    frame.setLayout(layout)
    return sliders_list, indicators


def clear(frame):
    layout = frame.layout()
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


def play_n_pause(button, timer_input, sound, player):
    def set_icon(is_playing):
        icon_path = "icons/pause.png" if is_playing else "icons/play.png"
        icon = QIcon(icon_path)
        button.setIcon(icon)

    if timer_input.isActive():
        timer_input.stop()
        set_icon(False)
        if sound:
            pause_audio(player)
    else:
        timer_input.start(100)
        set_icon(True)
        if sound:
            play_audio(player)


def speed(slider_value, speed_label, timer, button):
    map_speed_factor = {
        1: 0.2,
        2: 0.5,
        3: 1.0,
        4: 1.5,
        5: 2.0,
    }
    icon = QIcon("icons/pause.png")  # Use the resource path
    button.setIcon(icon)
    speed_factor = map_speed_factor[slider_value]
    speed_label.setText(f'x {speed_factor}')
    time_interval = int(100 / speed_factor)  # ms
    
    update_speed(timer, time_interval)


def update_speed(timer, time_interval):
    timer.stop()
    timer.start(time_interval)


def compute_fourier_transform(signal, Ts):
    fourier_transform = np.fft.rfft(signal)
    frequencies_fft = np.fft.rfftfreq(len(signal), Ts)
    amplitudes = np.abs(fourier_transform)
    phases = np.angle(fourier_transform)
    return amplitudes, frequencies_fft, phases


def apply_smoothing_window(output_amplitudes, index, parameter, plot_widget, start, end , freq_comp):
    if parameter == 0:
        parameter = 0.1
    window_functions = {
        0: scipy.signal.windows.hamming,
        1: scipy.signal.windows.hann,
        2: lambda length, std: scipy.signal.windows.gaussian(length, std * length),
        3: scipy.signal.windows.boxcar
    }
    # window_length = len(output_amplitudes)
    window_length = len(output_amplitudes[start:end])
    window_function = window_functions.get(index)
    output_range = output_amplitudes[start:end]
    scale = max(output_amplitudes)

    if window_function:
        if index == 2:  # Gaussian window requires extra parameter
            window = window_function(window_length, parameter)
        else:
            window = window_function(window_length)

        smoothed_signal = window * output_range
        output_amplitudes[start:end] = smoothed_signal
        frequency_domain_window = np.zeros_like(freq_comp)
        frequency_domain_window[start:end] = window
        plot_widget.clear()
        freq_domain_plotting(freq_comp, output_amplitudes, plot_widget)
        plot_widget.plot(freq_comp, scale * frequency_domain_window, pen='r', fillLevel=0, fillBrush=(255, 0, 0, 100))

        return smoothed_signal


def freq_domain_plotting(freq_comp, output_amplitudes, plot_widget):
    plot_widget.clear()
    plot_widget.plot(freq_comp, output_amplitudes, pen="b")
    plot_widget.setLabel('left', 'Amplitude')
    plot_widget.setLabel('bottom', 'Frequency (Hz)')


def compute_inverse_fourier_transform(new_amplitude, freq_comp, phases):
    new_amplitudes = new_amplitude
    new_fft_result = new_amplitudes * np.exp(1j * phases)
    inverse_fft = np.fft.irfft(new_fft_result)
    return inverse_fft


def play_audio(player):
    player.play()


def pause_audio(player):
    player.pause()


def plot_specto(data, sample_rate, frame, checkbox):
    clear(frame)
    if len(data) == 0:
        return
    else:
        if checkbox.isChecked():
            canvas = FigureCanvas(plt.Figure())
            layout = frame.layout()
            layout.addWidget(canvas)

            # Plot the spectrogram
            plt.figure()
            plt.specgram(data, Fs=sample_rate, cmap='viridis', aspect='auto')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectrogram')
            canvas.figure.clear()
            canvas.figure = plt.gcf()
            canvas.draw()
        else:
            return


def zoom(Input_plot_widget, output_plot_widget, zoom_factor):
    Input_plot_widget.getViewBox().scaleBy((zoom_factor, zoom_factor))
    output_plot_widget.getViewBox().scaleBy((zoom_factor, zoom_factor))


def plot_waveform(data, sample_rate, plot_widget):
    time = np.arange(0, len(data)) / sample_rate
    plot_widget.plot(time, data)
    plot_widget.setLabel('left', 'Amplitude')
    plot_widget.setLabel('bottom', 'Time (s)')
    plot_widget.showGrid(x=True, y=True)

    