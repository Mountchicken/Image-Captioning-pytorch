# Image-Captioning-pytorch
An Easy attempt to Image Captioning with Inception_V3 as backbone. Pytorch based, no attention used(May update latter)
- ![test_example](https://github.com/Mountchicken/Image-Captioning-pytorch/blob/main/text_examples/dog.jpg)
- Predicted: <SOS> a brown dog is running on the grass . <EOS>

## 项目结构
### 文件

- `Model.py`: 定义Inception_v3模型，LSTM模型
- `get_loader.py`:定义ImageCaptioning数据集
- `Train.py`: 训练模型，建议不要修改超参数，因为我发现好像只有特定的超参数才能有较好的训练效果
- `inferrence`: 测试你自己的图片


### 文件夹
- `archive`: 存放flickr8k数据集，[下载地址](https://www.kaggle.com/aladdinpersson/flickr8kimagescaptions)
- `test_examples`:测试图片
## 如何使用

### 如何训练
#### 1.下载spacy库所需文件
- `pip install spacy`
- `download en_core_web_sm,[download](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.0.0)
- `-pip install 安装包`
#### 2.运行train.py
### 如何测试自己的图片
- `修改inferrence中测试图片地址(23行),运行即可

## 联系方式（获取预训练权重）
- mountchicken@outlook.com


