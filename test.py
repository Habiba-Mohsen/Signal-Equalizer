import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

class AudioPlayerWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.player = QMediaPlayer()

        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.play_audio)
        layout.addWidget(self.play_button)

        self.pause_button = QPushButton('Pause', self)
        self.pause_button.clicked.connect(self.pause_audio)
        layout.addWidget(self.pause_button)

        self.file_label = QLabel('No file selected', self)
        layout.addWidget(self.file_label)

        self.select_file_button = QPushButton('Select File', self)
        self.select_file_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_file_button)

        self.setLayout(layout)
        self.setWindowTitle('Audio Player Example')
        self.setGeometry(100, 100, 300, 200)

    def play_audio(self):
        if self.player.state() == QMediaPlayer.StoppedState:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.file_path)))
        self.player.play()

    def pause_audio(self):
        self.player.pause()

    def select_file(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("WAV files (*.wav)")
        file_dialog.setDefaultSuffix("wav")

        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()[0]
            self.file_label.setText(f'Selected File: {self.file_path}')

def main():
    app = QApplication(sys.argv)
    window = AudioPlayerWidget()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
