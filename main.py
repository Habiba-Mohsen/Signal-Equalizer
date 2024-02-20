# File: main.py
from collections import namedtuple
import sys
from os import path
from PyQt5.uic import loadUiType
import functions as f
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import wave
import classes as c
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import sounddevice as sd
import scipy.signal
from scipy.signal import *
import csv


FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "design.ui"))


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)

        self.setupUi(self)
        self.setWindowTitle("Signal Equalizer")
        
        # Objects
        self.hamming = c.WindowType(['N'], 0)
        self.hanning = c.WindowType(['N'], 0)
        self.gaussian = c.WindowType(['Std'], 1)
        self.rectangle = c.WindowType(['constant'], 0)

        r = namedtuple('Range', ['min', 'max'])
        self.default = c.Mode([f'{i*10} to {(i+1)*10} Hz' for i in range(10)],
                              [r(i * 1000 + 1, (i + 1) * 1000) for i in range(10)], 10)
        self.ecg = c.Mode(['Normal ECG', 'A1', 'A2', 'A3'], [1]+[0 for _ in range(3)], 4)
        self.animals = c.Mode(['Wolf', 'Owl', 'Horse', 'Bat'], [1 for _ in range(4)], 4)
        self.musical = c.Mode(['Guitar', 'Flutes', 'Xylophone', 'Drums'], [1 for _ in range(4)], 4)

        # Variables
        
        self.index = 0
        self.state = True
        self.audio_data = []
        self.edited_time_domain_signal = []
        self.sample_rate = 44100
        self.playing = False
        self.sliders_list = []
        self.indicators_list = []
        self.window_sliders = []
        self.window_indicators = []
        self.mapping_mode = {
            0: self.default,
            1: self.ecg,
            2: self.animals,
            3: self.musical,
        }
        self.window_map = {
            0: self.hamming,
            1: self.hanning,
            2: self.gaussian,
            3: self.rectangle,
        }
        self.animals_mode = {0: [2754, 8261], 1: [8262,  13769], 2: [17901,42686], 3: [42687, 82619]}
        self.music_mode = {0: [5096, 50956], 1: [50957, 101913], 2: [101914, 152869], 3: [152870, 968176]}
        # self.ecg_mode = {0: [5, 190], 1: [1, 25], 2: [95, 249], 3:[141,190]}
        self.ecg_mode = {0: [5, 140], 1:[350,500] , 2: [95, 249], 3:[141,190]}
        self.default_mode = {0: [0, 50], 1: [50, 100], 2: [100, 150], 3: [150, 200], 4: [200, 250], 5: [250, 300], 6: [300, 350], 7: [350, 400], 8: [400, 450], 9: [450, 500]}

        # Timers
        self.timer_input = QtCore.QTimer()
        self.timer_output = QtCore.QTimer()
        self.timer1 = QtCore.QTimer()
        self.timer2 = QtCore.QTimer()
        self.timer_input.timeout.connect(lambda: self.update_waveform(self.audio_data, self.InputGraph))
        self.timer_input.timeout.connect(lambda: self.update_waveform(self.edited_time_domain_signal, self.OutputGraph))

        # Audio Players
        self.media_playerIN = QMediaPlayer()


        # Setting the Ui
        self.SliderFrame.setMaximumHeight(200)
        self.change_mode(self.mode_comboBox.currentIndex())
        self.smoothing_window_type(self.window_comboBox.currentIndex())
        self.InputGraph.setBackground('w')
        self.OutputGraph.setBackground('w')
        self.freqGraph.setBackground('w')

        # Signals
        self.importButton.clicked.connect(lambda: self.upload(self.musicfileName))
        self.mode_comboBox.currentIndexChanged.connect(lambda: self.change_mode(self.mode_comboBox.currentIndex()))
        self.window_comboBox.currentIndexChanged.connect(
            lambda: self.smoothing_window_type(self.window_comboBox.currentIndex()))
        self.playallButton.clicked.connect(lambda: f.play_n_pause(self.playallButton, self.timer_input, False, any))
        self.playButton1.clicked.connect(lambda: f.play_n_pause(self.playButton1, self.timer1, True, self.media_playerIN))
        self.playButton2.clicked.connect(lambda: self.play_output_signal(self.playButton2, self.edited_time_domain_signal, self.sample_rate))
        self.speedSlider.valueChanged.connect(lambda: f.speed(self.speedSlider.value(), self.speedLabel, self.timer_input, self.playallButton))
        self.resetButton.clicked.connect(self.reset)
        self.showCheckBox.stateChanged.connect(lambda: f.plot_specto(self.audio_data, self.sample_rate, self.spectoframe1, self.showCheckBox))
        self.showCheckBox.stateChanged.connect(lambda: f.plot_specto(self.edited_time_domain_signal, self.sample_rate, self.spectoframe2, self.showCheckBox))
        self.window_comboBox.currentIndexChanged.connect(lambda: self.get_smoothing_window(self.window_comboBox.currentIndex(), self.freqGraph, self.output_amplitudes, self.frequency_comp, 1))
        self.zoomInButton.clicked.connect(lambda: f.zoom(self.InputGraph, self.OutputGraph, 0.8 ))
        self.zoomOutButton.clicked.connect(lambda: f.zoom(self.InputGraph, self.OutputGraph, 1.2))
        self.SliderFrame.setEnabled(False)
        self.window_comboBox.setEnabled(False)
        self.window_comboBox.setCurrentIndex(3)

    # FUNCTIONS
    def enable_widgets(self):
        self.playallButton.setEnabled(True)
        self.resetButton.setEnabled(True)
        self.zoomOutButton.setEnabled(True)
        self.zoomInButton.setEnabled(True)
        self.speedSlider.setEnabled(True)
        self.showCheckBox.setEnabled(True)
        self.SliderFrame.setEnabled(True)
        self.window_comboBox.setEnabled(True)

    def upload(self, label):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        filters = "Audio and CSV Files (*.wav *.csv)"
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileNames()", "", filters, options=options)
        self.media_playerIN.setMedia(QMediaContent())
        if file_path:
            # Store file name
            file_name = file_path.split('/')[-1]
            label.setText(file_name)

            if file_path.lower().endswith('.wav'):
                if self.media_playerIN.state() == QMediaPlayer.StoppedState:
                        self.media_playerIN.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))

                # Open the .wav file for reading
                with wave.open(file_path, 'rb') as audio_file:   #reading file in binary mode
                    # Get the audio file's parameters
                    num_frames = audio_file.getnframes()

                    # Read audio data as bytes
                    raw_audio_data = audio_file.readframes(num_frames)

                    # Convert raw bytes to numerical values (assuming 16-bit PCM)
                    self.audio_data = np.frombuffer(raw_audio_data, dtype=np.int16)
                    self.edited_time_domain_signal = self.audio_data.copy()

                    self.sample_rate = audio_file.getframerate()
                    self.time = np.arange(0, len(self.audio_data)) / self.sample_rate

            elif file_path.lower().endswith('.csv'):
                data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(1,))
                self.audio_data = data[0:1000]
                self.edited_time_domain_signal = self.audio_data.copy()
                self.x = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0,))
                self.sample_rate = 1/(self.x[1]-self.x[0])
                self.time = self.x[0:1000]
        self.reset_sliders()
        self.update_signal(self.mode_comboBox.currentIndex())
        f.freq_domain_plotting(self.frequency_comp, self.output_amplitudes, self.freqGraph)
        self.InputGraph.clear()
        self.OutputGraph.clear()
        f.plot_waveform(self.audio_data, self.sample_rate, self.InputGraph)
        f.plot_waveform(self.edited_time_domain_signal, self.sample_rate, self.OutputGraph)
        self.enable_widgets()
        if self.showCheckBox.isChecked():
            f.plot_specto(self.audio_data, self.sample_rate, self.spectoframe1, self.showCheckBox)
            f.plot_specto(self.edited_time_domain_signal, self.sample_rate, self.spectoframe2, self.showCheckBox)

    def clear_and_plotwaveform(self):
        self.InputGraph.clear()
        self.OutputGraph.clear()
        self.index = 0
        self.timer_input.start()
        self.state = False

    def reset(self):
        if self.timer_input.isActive():
            f.play_n_pause(self.playallButton, self.timer_input, False, _)
            self.clear_and_plotwaveform()
        else:
            self.clear_and_plotwaveform()

    def update_waveform(self, data, plot_widget):
        if self.state:
            self.InputGraph.clear()
            self.OutputGraph.clear()

        self.state = False
        x_min = self.index
        x_max = min(len(self.time), self.index + 10)

        plot_item = plot_widget.plot(pen='b')
        plot_item.setData(self.time[x_min:x_max], data[x_min:x_max])
        plot_widget.setXRange(self.time[x_min], self.time[x_max])

        if self.index >= len(self.time):
            self.index = 0
        self.index += 1

    def change_mode(self, index):
        mode = self.mapping_mode[index]
        self.sliders_list, indicators_list = f.create_sliders(mode.num_sliders, mode.labels, self.SliderFrame, 2)
        self.sliders_refresh(self.sliders_list, indicators_list)
        self.update_signal(index)
        self.connect_slider_signals()
        self.window_comboBox.setCurrentIndex(3)
    def update_signal(self, index):
        self.signal = self.audio_data
        Ts = 1/self.sample_rate
        if len(self.signal):
            self.amplitudes, self.frequency_comp, self.phases = f.compute_fourier_transform(self.signal, Ts)
            self.output_amplitudes = self.amplitudes.copy()

    def sliders_refresh(self, sliders, indicators):
        if sliders:
            for slider in sliders:
               slider.valueChanged.connect(lambda: self.update_indicators(sliders, indicators))

    def update_indicators(self, sliders, indicators):
        if sliders:
            for i, slider in enumerate(sliders):
                indicators[i].setText(f"{slider.value()}")

    def smoothing_window_type(self, index):
        window = self.window_map[index]

        self.window_sliders, self.window_indicators = f.create_sliders(window.num_sliders, window.labels,
                                                                       self.WindowFrame, 1)

        # Refresh Sliders
        self.sliders_refresh(self.window_sliders, self.window_indicators)
        for slider in self.window_sliders: slider.valueChanged.connect(lambda: self.customize_smoothing_window_parameters(slider.value(), self.window_comboBox.currentIndex(), self.freqGraph, self.output_amplitudes, self.frequency_comp))

    def connect_slider_signals(self):   # connect sliders with modifying amplitudes
        for slider in self.sliders_list: slider.valueChanged.connect(lambda value, Slider=slider: self.modifying_amplitudes(self.sliders_list.index(Slider), Slider.value(), self.amplitudes, self.output_amplitudes, self.window_comboBox.currentIndex(), 1))

    def modify_output_amplitudes(self, mode_index, freq_component_index, gain, input_amplitudes, output_amplitudes,
                                 window_index, parameter, frequency_comp, freqGraph):
        # indice = np.where((self.frequency_comp > 50) & (self.frequency_comp <len(input_amplitudes)))[0]
        # print(indice)
        mode_ranges = {
            0: self.default_mode,
            1: self.ecg_mode,
            2: self.animals_mode,
            3: self.music_mode
        }

        start, end = mode_ranges[mode_index][freq_component_index]
        output_amplitudes[start:end] = gain* input_amplitudes[start:end]
        output_amplitudes[start:end] = f.apply_smoothing_window(output_amplitudes, window_index, parameter, freqGraph,start, end, frequency_comp)

        return output_amplitudes

    def modifying_amplitudes(self, freq_component_index, gain, input_amplitudes, output_amplitudes, window_index,
                             parameter):
        mode_index = self.mode_comboBox.currentIndex()

        output_amplitudes = self.modify_output_amplitudes(mode_index, freq_component_index, gain, input_amplitudes,
                                                     output_amplitudes, window_index, parameter, self.frequency_comp,
                                                     self.freqGraph)

        self.smooth_and_inverse_transform(output_amplitudes)


    def get_smoothing_window(self, window_index, plot_widget, output_amp, freq_comp, parameter):  #type of smoothing window
        self.modifying_amplitudes(0, 1, output_amp, output_amp, window_index, 1)

    def customize_smoothing_window_parameters(self, value, window_index, plot_widget, output_amp, freq_comp):    #smoothing window parameter
        new_value = value
        self.modifying_amplitudes(0, 1, output_amp, output_amp, window_index, new_value)

    def smooth_and_inverse_transform(self, output_amplitudes):
        self.edited_time_domain_signal = f.compute_inverse_fourier_transform(output_amplitudes, self.frequency_comp, self.phases)
        self.OutputGraph.clear()
        if self.state == True:
            f.plot_waveform(self.edited_time_domain_signal, self.sample_rate, self.OutputGraph)
        if self.showCheckBox.isChecked():
            f.plot_specto(self.edited_time_domain_signal, self.sample_rate, self.spectoframe2, self.showCheckBox)
        self.playing=True
        if self.mode_comboBox.currentIndex()==2 or self.mode_comboBox.currentIndex()==3:
            self.play_output_signal(self.playButton2,self.edited_time_domain_signal,self.sample_rate)
        # self.save_csv()

    def play_output_signal(self, button, samples, sample_rate):
        if self.playing:
            icon = QIcon("icons/play.png")
            button.setIcon(icon)
            sd.stop()
            self.playing = False
        else:
            icon = QIcon("icons/pause.png")
            button.setIcon(icon)
            new_samples = samples.astype(np.float32) / 32767.0  # Convert int16 to float32
            sd.play(new_samples, sample_rate)
            self.playing = True

    def reset_sliders(self):
        for slider in self.sliders_list:
            slider.setValue(1)

    # def save_csv(self):
    #         csv_file_path = r'C:\Users\hp\Documents\Arrythmia1.csv'
    #
    #         # Open the CSV file in write mode
    #         with open(csv_file_path, 'w', newline='') as csvfile:
    #             # Create a CSV writer object
    #             csv_writer = csv.writer(csvfile)
    #
    #             # Write header
    #             csv_writer.writerow(["Time", "Amp"])
    #
    #             # Write rows with time and edited time-domain signal values
    #             for time_value, signal_value in zip(self.time, self.edited_time_domain_signal.tolist()):
    #                 csv_writer.writerow([time_value, signal_value])

def main():
    app = QApplication(sys.argv)

    # with open("style.qss", "r") as f:
    #     _style = f.read()
    #     app.setStyleSheet(_style)

    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
