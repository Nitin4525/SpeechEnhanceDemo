import os
import time
import cpuinfo
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from scipy import signal

import torch

from PyQt5 import QtWidgets, QtMultimedia
from PyQt5.QtCore import QUrl, Qt

from mainWindow import Ui_MainWindow
from analysisWindow import Ui_analysisDialog

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


def slice_signal(file, window_size, hop, sample_rate):
    wav_pre, sr = librosa.load(file, sr=None)
    if sr == sample_rate:
        wav = wav_pre
    else:
        wav = librosa.resample(wav_pre, sr, sample_rate)
    wav = wav/np.max(wav)
    src_length = len(wav)
    padding_len = hop - ((len(wav) - window_size) % hop)
    wav = np.pad(wav, (0, padding_len), 'constant')
    slices = []
    for start_idx in range(0, len(wav)-window_size+1, hop):
        slice_sig = wav[start_idx:start_idx+window_size]
        slices.append(slice_sig)
    return slices, src_length


def connect_signal(sliceA, sliceB, overlap_length):
    sliceA_self = sliceA[:-overlap_length]
    sliceA_overlap = sliceA[-overlap_length:]
    sliceB_self = sliceB[overlap_length:]
    sliceB_overlap = sliceB[:overlap_length]

    right = len(sliceA_overlap)
    weight_A = []
    weight_B = []
    for i in range(right):
        weight_A.append(1-i / right)
        weight_B.append(i / right)
    overlap = sliceA_overlap * weight_A + sliceB_overlap * weight_B
    return np.concatenate((sliceA_self, overlap, sliceB_self))


def emphasis(signal_batch, emph_coeff=0.95, pre=True):
    if pre:
        return signal.lfilter([1, -emph_coeff], [1], signal_batch)
    else:
        return signal.lfilter([1], [1, -emph_coeff], signal_batch)


class showFigure(FigureCanvas):
    def __init__(self, width, height, dpi):
        self.figure = plt.figure(figsize=(width, height), dpi=dpi)
        super(showFigure, self).__init__(self.figure)


class analysis(QtWidgets.QDialog):
    def __init__(self, windowTitle):
        super(analysis, self).__init__()
        self.player = QtMultimedia.QMediaPlayer()
        ui = Ui_analysisDialog()
        self.setWindowTitle(windowTitle)
        ui.setupUi(self)

        filenames = windowTitle.split('->')
        ui.pushButton_noise.clicked.connect(lambda: self.playAudio(filenames[0]))
        figure_noise = showFigure(5, 4, 100)
        figure_noise.figure.add_subplot(2, 1, 1)
        wav, sr = librosa.load(filenames[0], sr=None)
        librosa.display.waveplot(wav, sr)
        figure_noise.figure.add_subplot(2, 1, 2)
        librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=512, hop_length=512), ref=np.max), sr=sr, x_axis='time', y_axis='linear')
        plt.xlabel('Time/s')
        plt.ylabel('Frequency/Hz')
        plt.tight_layout()
        scence_noise = QtWidgets.QGraphicsScene()
        scence_noise.addWidget(figure_noise)
        ui.graphicsView_noise.setScene(scence_noise)
        ui.graphicsView_noise.show()
        if len(filenames) == 2:
            ui.pushButton_enhanced.clicked.connect(lambda: self.playAudio(filenames[1]))
            figure_enhanced = showFigure(5, 4, 100)
            figure_enhanced.figure.add_subplot(2, 1, 1)
            wav, sr = librosa.load(filenames[1], sr=None)
            librosa.display.waveplot(wav, sr)
            figure_enhanced.figure.add_subplot(2, 1, 2)
            librosa.display.specshow(
                librosa.power_to_db(librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=512, hop_length=512),
                                    ref=np.max), sr=sr, x_axis='time', y_axis='linear')
            plt.xlabel('Time/s')
            plt.ylabel('Frequency/Hz')
            plt.tight_layout()
            scence_enhanced = QtWidgets.QGraphicsScene()
            scence_enhanced.addWidget(figure_enhanced)
            ui.graphicsView_enhanced.setScene(scence_enhanced)
            ui.graphicsView_enhanced.show()

    def playAudio(self, filename):
        if os.access(filename, os.R_OK):
            file = QUrl.fromLocalFile(filename)
            content = QtMultimedia.QMediaContent(file)
            self.player.setMedia(content)
            self.player.setVolume(50)
            self.player.play()

    def closeEvent(self, event):
        self.player.stop()
        event.accept()


class Demo(QtWidgets.QMainWindow):
    def __init__(self):
        super(Demo, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 初始化界面
        self.ui.radioButton_cpu.setChecked(True)

        # 初始化模型
        self.model = None
        self.iscuda = torch.cuda.is_available()
        if not self.iscuda:
            self.ui.radioButton_gpu.setEnabled(False)
        self.device = 'cpu'
        self.model_type = 'SEGAN'
        model_init_flag = self.loadModel()
        self.detectSystem()

        self.ui.textBrowser_log.append('{} 初始化完成'.format(time.strftime("%H:%M:%S", time.localtime())))
        if model_init_flag:
            self.ui.textBrowser_log.append('{} 初始化模型为SEGAN'.format(time.strftime("%H:%M:%S", time.localtime())))
            self.ui.radioButton_segan.setChecked(True)
        else:
            self.ui.textBrowser_log.append('{} 初始化模型失败，请检查模型文件路径'.format(time.strftime("%H:%M:%S", time.localtime())))
            self.ui.groupBox_model.setEnabled(False)
        self.ui.textBrowser_log.append('{} 初始化推理设备为CPU'.format(time.strftime("%H:%M:%S", time.localtime())))

        # 信号槽
        self.ui.pushButton_readfiles.clicked.connect(self.readFiles)
        self.ui.pushButton_changepath_read.clicked.connect(lambda: self.changePath(self.ui.pushButton_changepath_read))
        self.ui.pushButton_changepath_save.clicked.connect(lambda: self.changePath(self.ui.pushButton_changepath_save))
        self.ui.pushButton_changepath_mod.clicked.connect(lambda: self.changePath(self.ui.pushButton_changepath_mod))
        # self.ui.pushButton_changepath_pic_3.clicked.connect(lambda: self.changePath(self.ui.pushButton_changepath_pic_3))
        self.ui.pushButton_clearlist.clicked.connect(self.clearList)
        self.ui.pushButton_clearlog.clicked.connect(self.clearLogs)
        self.ui.pushButton_forward.clicked.connect(self.doForward)
        self.ui.pushButton_hardware.clicked.connect(self.detectSystem)
        self.ui.radioButton_gpu.toggled.connect(lambda: self.selectDevice(self.ui.radioButton_gpu))
        self.ui.radioButton_cpu.toggled.connect(lambda: self.selectDevice(self.ui.radioButton_cpu))
        self.ui.radioButton_segan.toggled.connect(lambda: self.selectModel(self.ui.radioButton_segan))
        self.ui.radioButton_tasnet.toggled.connect(lambda: self.selectModel(self.ui.radioButton_tasnet))
        self.ui.pushButton_delfiles.clicked.connect(self.removeItem)
        self.ui.pushButton_analysis.clicked.connect(self.analysisAudio)

    def readFiles(self):
        filepaths, filetype = QtWidgets.QFileDialog.getOpenFileNames(self, "选取文件", self.ui.label_readpath.text(), 'Audio Files(*.wav)')
        if filepaths:
            for filepath in filepaths:
                if self.ui.listWidget_process.findItems(filepath, Qt.MatchExactly):
                    self.ui.textBrowser_log.append('{} 文件{}已存在'.format(time.strftime("%H:%M:%S", time.localtime()), filepath))
                else:
                    self.ui.listWidget_process.addItem(filepath)
                    self.ui.textBrowser_log.append('{} 添加文件{}到处理队列'.format(time.strftime("%H:%M:%S", time.localtime()), filepath))
            self.ui.textBrowser_log.append('{} 当前队列中文件数为： {}'.format(time.strftime("%H:%M:%S", time.localtime()), self.ui.listWidget_process.count()))
        else:
            self.ui.textBrowser_log.append('{} 未选中文件'.format(time.strftime("%H:%M:%S", time.localtime())))

    def selectModel(self, button):
        if button.isChecked():
            self.model_type = button.text()
            model_flag = self.loadModel()
            if model_flag:
                self.ui.textBrowser_log.append('{} 模型重载为{}'.format(time.strftime("%H:%M:%S", time.localtime()), button.text()))
            else:
                self.ui.textBrowser_log.append('{} 模型重载失败，检查模型路径'.format(time.strftime("%H:%M:%S", time.localtime())))

    def selectDevice(self, button):
        if button.isChecked():
            self.device = 'cpu' if button.text() == 'CPU' else 'cuda'
            self.model.to(self.device)
            self.ui.textBrowser_log.append('{} 推理设备变更为{}'.format(time.strftime("%H:%M:%S", time.localtime()), button.text()))

    def doForward(self):
        if self.model is not None:
            if os.path.exists(self.ui.label_savepath.text()):
                length = self.ui.listWidget_process.count()
                if length:
                    for i in range(length):
                        item = self.ui.listWidget_process.item(i)
                        if len(item.text().split('->')) == 2:
                            continue
                        with torch.no_grad():
                            noisy_slices, src_length = slice_signal(item.text(), 16384, 8192, 16000)
                            enhanced_speech = []
                            for noisy_slice in noisy_slices:
                                noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(
                                    torch.FloatTensor).to(self.device)
                                if self.model_type == 'SEGAN':
                                    z = torch.nn.init.normal_(torch.Tensor(1, 1024, 8)).to(self.device)
                                    generated_speech = self.model(noisy_slice, z).data.cpu().numpy()
                                elif self.model_type == 'TASNET':
                                    generated_speech = self.model(noisy_slice).data.cpu().numpy()
                                generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
                                generated_speech = generated_speech.reshape(-1)
                                enhanced_speech.append(generated_speech)
                        enhanced_speech_ = enhanced_speech[0]
                        for j in range(1, len(enhanced_speech)):
                            enhanced_speech_ = connect_signal(enhanced_speech_, enhanced_speech[j], overlap_length=8192)
                        file_name = os.path.join(self.ui.label_savepath.text(), '{}.wav'.format(os.path.basename(item.text()).split('.')[0]))
                        while os.path.exists(file_name):
                            file_name = file_name.split('.')[0] + '_' + '.wav'
                        self.ui.textBrowser_log.append('{} {}/{} 保存结果于{}'.format(time.strftime("%H:%M:%S", time.localtime()), i + 1, length, file_name))
                        self.ui.listWidget_process.openPersistentEditor(item)
                        item.setText(item.text()+'->'+file_name)
                        self.ui.listWidget_process.closePersistentEditor(item)
                        sf.write(file_name, enhanced_speech_.T[:src_length], 16000, )
                    self.ui.textBrowser_log.append('{} 文件处理完毕，请重新装入待处理文件'.format(time.strftime("%H:%M:%S", time.localtime())))
                else:
                    self.ui.textBrowser_log.append('{} 处理队列为空, 请选择文件'.format(time.strftime("%H:%M:%S", time.localtime())))
            else:
                self.ui.textBrowser_log.append('{} 请检查文件保存路径'.format(time.strftime("%H:%M:%S", time.localtime())))
        else:
            self.ui.textBrowser_log.append('{} 模型未初始化，请选择模型'.format(time.strftime("%H:%M:%S", time.localtime())))

    def clearList(self):
        self.ui.listWidget_process.clear()
        self.ui.textBrowser_log.append('{} 清空处理队列'.format(time.strftime("%H:%M:%S", time.localtime())))
        self.ui.textBrowser_log.append('{} 当前队列中文件数为： {}'.format(time.strftime("%H:%M:%S", time.localtime()), self.ui.listWidget_process.count()))

    def loadModel(self):
        model_name = r'{}.pt'.format(self.model_type)
        model_path = os.path.join(self.ui.label_modelpath.text(), model_name)
        if os.access(model_path, os.R_OK):
            self.model = torch.jit.load(model_path, map_location=self.device)
            return True
        else:
            return False

    def detectSystem(self):
        self.ui.label_cpu.setText(cpuinfo.get_cpu_info()['brand_raw'])
        if self.iscuda:
            self.ui.label_gpu.setText(torch.cuda.get_device_name(0))
            self.ui.label_cuda.setText(torch.version.cuda)
        else:
            self.ui.label_gpu.setText('不可用')
            self.ui.label_cuda.setText('不可用')
        self.ui.label_pytorch.setText(torch.__version__)

    def clearLogs(self):
        self.ui.textBrowser_log.clear()

    def removeItem(self):
        item = self.ui.listWidget_process.currentItem()
        if item:
            self.ui.listWidget_process.takeItem(self.ui.listWidget_process.row(item))
            self.ui.textBrowser_log.append('{} 移除文件{}'.format(time.strftime("%H:%M:%S", time.localtime()), item.text()))
            self.ui.textBrowser_log.append('{} 当前队列中文件数为： {}'.format(time.strftime("%H:%M:%S", time.localtime()), self.ui.listWidget_process.count()))
        else:
            self.ui.textBrowser_log.append('{} 未选中文件'.format(time.strftime("%H:%M:%S", time.localtime())))

    def changePath(self, button):
        filePath = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件夹", r"./")
        if filePath:
            button_name = button.accessibleName()
            if button_name == 'changeread':
                self.ui.label_readpath.setText(filePath)
                self.ui.textBrowser_log.append('{} 变更{}路径为{}'.format(time.strftime("%H:%M:%S", time.localtime()), '文件读取', filePath))
            elif button_name == 'changemodel':
                self.ui.label_modelpath.setText(filePath)
                self.ui.textBrowser_log.append('{} 变更{}路径为{}'.format(time.strftime("%H:%M:%S", time.localtime()), '模型读取', filePath))
                self.ui.groupBox_model.setEnabled(True)
            elif button_name == 'changesave':
                self.ui.label_savepath.setText(filePath)
                self.ui.textBrowser_log.append('{} 变更{}路径为{}'.format(time.strftime("%H:%M:%S", time.localtime()), '结果保存', filePath))
            # elif button_name == 'changepic':
            #     self.ui.label_picpath.setText(filePath)
            #     self.ui.textBrowser_log.append('{} 变更{}路径为{}'.format(time.strftime("%H:%M:%S", time.localtime()), '图片保存', filePath))
        else:
            pass

    def analysisAudio(self):
        item = self.ui.listWidget_process.currentItem()
        if item:
            analysisWin = analysis(item.text())
            analysisWin.show()
            analysisWin.exec_()
        else:
            self.ui.textBrowser_log.append('{} 未选中文件'.format(time.strftime("%H:%M:%S", time.localtime())))

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, '警告', '确认退出?', QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = Demo()
    win.show()
    sys.exit(app.exec())
    pass