# SpeechEnhancementDemo

#### 介绍
一个简单的语音增强Demo  
模型源码和训练见https://github.com/Nitin4525/SpeechEnhancement  
C++打包版之后补充

#### 软件功能
音频文件去噪

#### 安装教程
1.  `conda create -n SpeechEnhancementDemo`  # 创建python环境
2.  `conda install pytorch==1.8.1 cudatoolkit=10.1 torchaudio -c pytorch`  # 安装pytorch，注意此为cuda版本，cpu版本见Pytorch网站
3.  `pip install -r requirements.txt`  # 安装第三方包

#### 使用说明

1.  `git clone https://gitee.com/Nitin525/speech-enhancement-demo.git`
2.  `cd speech-enhancement-demo/`
3.  `conda activate SpeechEnhancementDemo`
4.  `python main.py`

#### 注意
由于没有加入启动动画，启动时若加载时间过长，窗口加载过程表现为无动作，若无控制台报错请等待主窗口出现。

