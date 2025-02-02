# DigitalFilm 数字胶卷 - kodark gold 200

DigitalFilm：使用一个神经网络来模拟胶卷风格。

---

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="./readme.md">
    <img src="images/logo.svg" alt="Logo" width="320" height="160">
  </a>

  <h3 align="center">"DigitalFilm 数字胶卷</h3>
  <p align="center">
    使用一个神经网络来模拟胶卷风格。
    <br />
    <a href="https://github.com/shaojintian/Best_README_template"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="./app/digitalFilm.py">查看Demo</a>
    ·
    <a href="">报告Bug</a>
    ·
    <a href="">提出新特性</a>
  </p>

</p>


 本篇README.md面向开发者和用户
 
## 目录

- [DigitalFilm 数字胶卷 - kodark gold 200](#digitalfilm-数字胶卷---kodark-gold-200)
  - [目录](#目录)
    - [运行 Demo](#运行-demo)
          - [**安装步骤**](#安装步骤)
    - [整体架构](#整体架构)
    - [数据集](#数据集)
    - [生成的图片对比](#生成的图片对比)
    - [生成图片的色彩空间](#生成图片的色彩空间)
    - [文件目录说明](#文件目录说明)
    - [版本控制](#版本控制)
    - [作者](#作者)
    - [版权说明](#版权说明)

### 运行 Demo

```bash
python digitalFilm.py [-v/-h/-g] -i <input> -o <ouput> -m <model>
```
- -v 打印版本信息
- -h 帮助信息
- -g 图形化选择图片
- -i 输入图片的目录
- -o 输出图片的目录
- -m 模型目录

###### **安装步骤**

```sh
git clone https://github.com/SongZihui-sudo/digitalFilm.git
```

最好现在conda里创建好环境，然后安装各种依赖。

```sh
pip install -r requirement.txt
```

### 整体架构

整体的架构，首先通过人工标注的数码-模拟胶片图像对来训练数据生成器，然后在使用生成器生成数码标签。最后通过数码-模拟胶片图像进行与训练，然后在使用生成数码-真实胶片照片数据集进行微调模型。

```txt
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 200, 320]             896
       BatchNorm2d-2         [-1, 32, 200, 320]              64
         LeakyReLU-3         [-1, 32, 200, 320]               0
            Conv2d-4         [-1, 64, 200, 320]          18,496
       BatchNorm2d-5         [-1, 64, 200, 320]             128
         LeakyReLU-6         [-1, 64, 200, 320]               0
 AdaptiveAvgPool2d-7             [-1, 64, 1, 1]               0
            Conv2d-8              [-1, 4, 1, 1]             256
              ReLU-9              [-1, 4, 1, 1]               0
           Conv2d-10             [-1, 64, 1, 1]             256
AdaptiveMaxPool2d-11             [-1, 64, 1, 1]               0
           Conv2d-12              [-1, 4, 1, 1]             256
             ReLU-13              [-1, 4, 1, 1]               0
           Conv2d-14             [-1, 64, 1, 1]             256
          Sigmoid-15             [-1, 64, 1, 1]               0
 ChannelAttention-16             [-1, 64, 1, 1]               0
           Conv2d-17            [-1, 128, 1, 1]          73,856
      BatchNorm2d-18            [-1, 128, 1, 1]             256
        LeakyReLU-19            [-1, 128, 1, 1]               0
AdaptiveAvgPool2d-20            [-1, 128, 1, 1]               0
           Conv2d-21              [-1, 8, 1, 1]           1,024
             ReLU-22              [-1, 8, 1, 1]               0
           Conv2d-23            [-1, 128, 1, 1]           1,024
AdaptiveMaxPool2d-24            [-1, 128, 1, 1]               0
           Conv2d-25              [-1, 8, 1, 1]           1,024
             ReLU-26              [-1, 8, 1, 1]               0
           Conv2d-27            [-1, 128, 1, 1]           1,024
          Sigmoid-28            [-1, 128, 1, 1]               0
 ChannelAttention-29            [-1, 128, 1, 1]               0
           Conv2d-30            [-1, 256, 1, 1]         295,168
      BatchNorm2d-31            [-1, 256, 1, 1]             512
        LeakyReLU-32            [-1, 256, 1, 1]               0
AdaptiveAvgPool2d-33            [-1, 256, 1, 1]               0
           Conv2d-34             [-1, 16, 1, 1]           4,096
             ReLU-35             [-1, 16, 1, 1]               0
           Conv2d-36            [-1, 256, 1, 1]           4,096
AdaptiveMaxPool2d-37            [-1, 256, 1, 1]               0
           Conv2d-38             [-1, 16, 1, 1]           4,096
             ReLU-39             [-1, 16, 1, 1]               0
           Conv2d-40            [-1, 256, 1, 1]           4,096
          Sigmoid-41            [-1, 256, 1, 1]               0
 ChannelAttention-42            [-1, 256, 1, 1]               0
           Conv2d-43            [-1, 128, 1, 1]         295,040
      BatchNorm2d-44            [-1, 128, 1, 1]             256
        LeakyReLU-45            [-1, 128, 1, 1]               0
AdaptiveAvgPool2d-46            [-1, 128, 1, 1]               0
           Conv2d-47              [-1, 8, 1, 1]           1,024
             ReLU-48              [-1, 8, 1, 1]               0
           Conv2d-49            [-1, 128, 1, 1]           1,024
AdaptiveMaxPool2d-50            [-1, 128, 1, 1]               0
           Conv2d-51              [-1, 8, 1, 1]           1,024
             ReLU-52              [-1, 8, 1, 1]               0
           Conv2d-53            [-1, 128, 1, 1]           1,024
          Sigmoid-54            [-1, 128, 1, 1]               0
 ChannelAttention-55            [-1, 128, 1, 1]               0
           Conv2d-56             [-1, 64, 1, 1]          73,792
      BatchNorm2d-57             [-1, 64, 1, 1]             128
        LeakyReLU-58             [-1, 64, 1, 1]               0
AdaptiveAvgPool2d-59             [-1, 64, 1, 1]               0
           Conv2d-60              [-1, 4, 1, 1]             256
             ReLU-61              [-1, 4, 1, 1]               0
           Conv2d-62             [-1, 64, 1, 1]             256
AdaptiveMaxPool2d-63             [-1, 64, 1, 1]               0
           Conv2d-64              [-1, 4, 1, 1]             256
             ReLU-65              [-1, 4, 1, 1]               0
           Conv2d-66             [-1, 64, 1, 1]             256
          Sigmoid-67             [-1, 64, 1, 1]               0
 ChannelAttention-68             [-1, 64, 1, 1]               0
           Conv2d-69             [-1, 32, 1, 1]          18,464
      BatchNorm2d-70             [-1, 32, 1, 1]              64
        LeakyReLU-71             [-1, 32, 1, 1]               0
AdaptiveAvgPool2d-72             [-1, 32, 1, 1]               0
           Conv2d-73              [-1, 2, 1, 1]              64
             ReLU-74              [-1, 2, 1, 1]               0
           Conv2d-75             [-1, 32, 1, 1]              64
AdaptiveMaxPool2d-76             [-1, 32, 1, 1]               0
           Conv2d-77              [-1, 2, 1, 1]              64
             ReLU-78              [-1, 2, 1, 1]               0
           Conv2d-79             [-1, 32, 1, 1]              64
          Sigmoid-80             [-1, 32, 1, 1]               0
 ChannelAttention-81             [-1, 32, 1, 1]               0
           Conv2d-82              [-1, 3, 1, 1]             867
           Conv2d-83          [-1, 3, 200, 320]              99
           Conv2d-84              [-1, 3, 1, 1]             195
================================================================
Total params: 805,161
Trainable params: 805,161
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.73
Forward/backward pass size (MB): 142.14
Params size (MB): 3.07
Estimated Total Size (MB): 145.94
----------------------------------------------------------------
```

### 数据集

在制作数据集这一阶段。首先基于数据增强技术，利用 Fimo 应用的数字滤镜生成仿真胶片效果图像，构建初步训练集进行模型预训练，同步开发数码照片标签生成模型。在第二阶段，针对真实场景优化模型性能，通过采集网络公开的柯达金200胶片样本构建数据集，运用前期训练的标签生成模型自动创建对应数码标签。最终采用两阶段训练框架：先基于仿真数据进行模型初始化，再通过真实样本数据集进行微调优化，这种渐进式训练方法有效提升了模型对真实影像特征的捕捉能力，同时缓解了原始数据不足带来的过拟合风险。

数据集由双源影像数据构成，主体部分采集自小米13 Ultra 手机拍摄的高质量数码照片，其余选自专业HDR影像数据集[1]。数据标注体系包含两个平行维度：(1)人工标注组：1517对精准配准的数码-模拟胶片图像对，其中胶片效果通过Fimo专业滤镜实现；(2)自动生成组：363张柯达金200专业胶片样片及其对应数码标签，通过预训练模型自动生成。在数据预处理阶段，采用动态数据增广策略，对输入图像实时施加随机空间变换，有效提升模型几何不变性。数据集按分层抽样原则进行划分，80%（1517×0.8+363×0.8=1504张）作为训练集，20%（376张）作为测试集，确保数码/胶片样本在训练测试集中的分布一致性。

### 生成的图片对比

![图a](./images/image_1.png)  

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">图1 对比</center> 


![alt text](./images/image_2.png) 
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">图2 对比</center> 

![alt text](./images/image_3.png)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">图3 对比</center> 


### 生成图片的色彩空间

![alt text](./images/image_5.png)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">图4 上图1的 R G B 通道</center> 


![alt text](./images/image_4.png)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">图4 上图1的色彩空间</center> 

### 文件目录说明

- DigitalFilm.ipynb 用来训练模型
- app   一个 Demo
  - digitalFilm.py 
  - mynet.py
  - kodark_gold_200.pt

### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

### 作者

151122876@qq.com SongZihui-sudo

知乎:Dr.who  &ensp; qq:1751122876    

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

### 版权说明

该项目签署了 GPLv3 授权许可，详情请参阅 [LICENSE.txt](./LICENSE.txt)