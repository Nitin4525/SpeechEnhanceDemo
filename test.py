# import torch
#
# # with torch.no_grad():
# #     model = torch.jit.load('model/SEGAN.pt')
# #     input = torch.rand([1, 1, 16384], dtype=torch.float32)
# #     z = torch.rand([1, 1024, 8], dtype=torch.float32)
# #
# #
# #     inputs = {input, z}
# #
# #     output = model(*inputs)
# #
# #     # inputs = {input}
# #     #
# #     # model = torch.jit.load('model/TASNET.pt')
# #     # output = model(*inputs)
# #
# # pass
# # import cpuinfo
# # print(cpuinfo.get_cpu_info()['brand_raw'])
# flag = torch.cuda.is_available()
# print(flag)

from PyQt5 import QtMultimedia
from PyQt5.QtCore import QUrl
import time  # 需要导入时间模块设置延时
#
# file = QUrl.fromLocalFile(r'D:\DEMAND\test\clean\p232_001.wav')  # 音频文件路径
# content = QtMultimedia.QMediaContent(file)
# player = QtMultimedia.QMediaPlayer()
# player.setMedia(content)
# player.setVolume(50)
# player.play()
item = 'dsa->`dasda'
print(len(item.split('->')))

