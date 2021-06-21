import os
import time
import cpuinfo
import librosa
import numpy as np
import soundfile as sf
from scipy import signal

import torch

from PyQt5 import QtWidgets

from mainWindow import Ui_MainWindow
from aboutWindow import Ui_aboutWindow


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


class Demo(QtWidgets.QMainWindow):
    def __init__(self):
        super(Demo, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 信号槽
        self.ui.actionAbout.triggered.connect(self.click_actionAbout)
        self.ui.toolButton_readfiles.clicked.connect(self.click_toolButton)
        self.ui.pushButton_clearall.clicked.connect(self.click_clearlist)
        self.ui.pushButton_clearlog.clicked.connect(self.click_clearlogs)
        self.ui.pushButton_forward.clicked.connect(self.click_forward)
        self.ui.pushButton_hardware.clicked.connect(self.hardware_detection)
        self.ui.radioButton_gpu.toggled.connect(lambda: self.select_device(self.ui.radioButton_gpu))
        self.ui.radioButton_cpu.toggled.connect(lambda: self.select_device(self.ui.radioButton_cpu))
        self.ui.radioButton_segan.toggled.connect(lambda: self.select_model(self.ui.radioButton_segan))
        self.ui.radioButton_tasnet.toggled.connect(lambda: self.select_model(self.ui.radioButton_tasnet))

        # 初始化变量
        self.process_list = []

        # 初始化模型
        self.model = None
        self.iscuda = torch.cuda.is_available()
        if not self.iscuda:
            self.ui.radioButton_gpu.setEnabled(False)
        self.device = 'cpu'
        self.model_type = 'SEGAN'
        self.load_model()

    def click_actionAbout(self):
        dialog = QtWidgets.QDialog()
        about_ui = Ui_aboutWindow()
        about_ui.setupUi(dialog)
        dialog.show()
        dialog.exec_()

    def click_toolButton(self):
        filePath, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", r"./", 'Audio Files(*.wav)')
        if filePath:
            if filePath in self.process_list:
                self.ui.textBrowser_log.append('{} 文件已存在'.format(time.strftime("%H:%M:%S", time.localtime())))
            else:
                self.process_list.append(filePath)
                self.ui.textBrowser_log.append('{} 添加文件{}到处理队列'.format(time.strftime("%H:%M:%S", time.localtime()), filePath))
                self.ui.textBrowser_log.append('{} 当前队列中文件数为： {}'.format(time.strftime("%H:%M:%S", time.localtime()), len(self.process_list)))
        else:
            self.ui.textBrowser_log.append('{} 未选中文件'.format(time.strftime("%H:%M:%S", time.localtime())))

    def select_model(self, button):
        if button.isChecked():
            self.model_type = button.text()
            self.load_model()
            self.ui.textBrowser_log.append('{} 模型重载为{}'.format(time.strftime("%H:%M:%S", time.localtime()), button.text()))

    def select_device(self, button):
        if button.isChecked():
            self.device = 'cpu' if button.text() == 'CPU' else 'cuda'
            self.model.to(self.device)
            self.ui.textBrowser_log.append('{} 推理设备变更为{}'.format(time.strftime("%H:%M:%S", time.localtime()), button.text()))

    def click_forward(self):
        if self.process_list:
            length = len(self.process_list)
            self.ui.textBrowser_log.append('{} 准备处理，文件数为： {}'.format(time.strftime("%H:%M:%S", time.localtime()), length))
            for idx, filename in enumerate(self.process_list):
                self.ui.textBrowser_log.append('{} {}/{} 预处理...'.format(time.strftime("%H:%M:%S", time.localtime()), idx+1, length))
                with torch.no_grad():
                    noisy_slices, src_length = slice_signal(filename, 16384, 8192, 16000)
                    enhanced_speech = []

                    self.ui.textBrowser_log.append('{} {}/{} 模型推理...'.format(time.strftime("%H:%M:%S", time.localtime()), idx + 1, length))
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
                for i in range(1, len(enhanced_speech)):
                    enhanced_speech_ = connect_signal(enhanced_speech_, enhanced_speech[i], overlap_length=8192)
                file_name = os.path.join('res', '{}.wav'.format(os.path.basename(filename).split('.')[0]))
                self.ui.textBrowser_log.append('{} {}/{} 保存结果于{}'.format(time.strftime("%H:%M:%S", time.localtime()), idx + 1, length, file_name))
                sf.write(file_name, enhanced_speech_.T[:src_length], 16000, )
            self.process_list.clear()
            self.ui.textBrowser_log.append('{} 文件处理完毕，请重新装入待处理文件'.format(time.strftime("%H:%M:%S", time.localtime())))
        else:
            self.ui.textBrowser_log.append('{} 处理队列为空'.format(time.strftime("%H:%M:%S", time.localtime())))
            self.ui.textBrowser_log.append('{} 请选择文件'.format(time.strftime("%H:%M:%S", time.localtime())))

    def click_clearlist(self):
        self.process_list.clear()
        self.ui.textBrowser_log.append('{} 清空处理队列'.format(time.strftime("%H:%M:%S", time.localtime())))
        self.ui.textBrowser_log.append('{} 当前队列中文件数为： {}'.format(time.strftime("%H:%M:%S", time.localtime()), len(self.process_list)))

    def load_model(self):
        model_path = r'model/{}.pt'.format(self.model_type)
        self.model = torch.jit.load(model_path, map_location=self.device)

    def hardware_detection(self):
        self.ui.label_cpu.setText(cpuinfo.get_cpu_info()['brand_raw'])
        if self.iscuda:
            self.ui.label_gpu.setText(torch.cuda.get_device_name(0))
            self.ui.label_cuda.setText(torch.version.cuda)
        else:
            self.ui.label_gpu.setText('不可用')
            self.ui.label_cuda.setText('不可用')
        self.ui.label_pytorch.setText(torch.__version__)

    def click_clearlogs(self):
        self.ui.textBrowser_log.clear()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = Demo()
    win.show()
    sys.exit(app.exec())
    pass