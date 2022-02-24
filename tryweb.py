from PyQt5 import QtCore, QtGui, QtNetwork
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon, QFont, QImage, QPixmap
from PyQt5.QtCore import QDir, Qt, QUrl, QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, 
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar)
# import controller
from PIL import Image, ImageQt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QBoxLayout, QFileDialog, QHBoxLayout, QLabel, QStatusBar, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPalette, QPixmap
from PyQt5.QtWidgets import QApplication, QPushButton
import sys
class IPWebcam(QtCore.QObject):
    pixmapChanged = QtCore.pyqtSignal(QtGui.QPixmap)
    
    def __init__(self, url, parent=None):
        super(IPWebcam, self).__init__(parent)
        self._url = url
        self.m_manager = QtNetwork.QNetworkAccessManager(self)
        self.m_manager.finished.connect(self._on_finished)

        self.m_stopped = True

    def start(self):
        self.m_stopped = False
        self._launch_request()

    def stop(self):
        self.m_stopped = True

    def _launch_request(self):
        request = QtNetwork.QNetworkRequest(QtCore.QUrl(self._url))
        self.m_manager.get(request)

    @QtCore.pyqtSlot(QtNetwork.QNetworkReply)
    def _on_finished(self, reply):
        ba = reply.readAll()
        pixmap = QtGui.QPixmap()
        if pixmap.loadFromData(ba):
            self.pixmapChanged.emit(pixmap)
        if not self.m_stopped:
            self._launch_request()


class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)

        self.m_label = QLabel()
        self.m_button = QPushButton(
            "Start", clicked=self.onClicked, checkable=True
        )

        lay = QVBoxLayout(self)
        lay.addWidget(self.m_label)
        lay.addWidget(self.m_button)

        self.resize(640, 480)

        url = "http://192.168.1.4:8080/shot.jpg"

        self.m_webcam = IPWebcam(url, self)
        self.m_webcam.pixmapChanged.connect(self.m_label.setPixmap)


    @QtCore.pyqtSlot(bool)
    def onClicked(self, checked):
        if checked:
            self.m_button.setText("Stop")
            self.m_webcam.start()
        else:
            self.m_button.setText("Start")
            self.m_webcam.stop()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())