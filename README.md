# stable-diffusion-GUI
PyQt5实现一个GUI界面，更加方便使用SD
## Stable Diffusion模型 + GUI界面（pytorch）
---

## Top News
**`2024-09`**:**GUI界面实现生成过程**  

## 性能情况
参考stable-diffusion的论文哈。  
https://ommer-lab.com/research/latent-diffusion-models/

## 所需环境
torch==2.0.1   
推荐torch==2.0.1，大多stable diffusion**基于这个版本**，webui也是。
```
实现参考https://blog.csdn.net/weixin_44791964/article/details/130588215    向大佬敬礼！！！
SD模型来源：https://github.com/bubbliiiing/stable-diffusion
我只是做了GUI界面，更方便使用！！！

# 安装torch==2.0.1 
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 安装其它requirement
pip install -r requirements.txt
# 为了加速可安装xformers。
pip install xformers==0.0.20
```

## 文件下载
训练所需的权值可在百度网盘中下载。  
链接: https://pan.baidu.com/s/1p4e1-jcJJt3lCFZeMpYbwA    
提取码: vry7     
  
Flickr8k数据集也可以在百度网盘中下载。  
链接：https://pan.baidu.com/s/1I2FfEOhcBOupUazJP18ADQ    
提取码：lx57   
训练需要较高的显存要求，需要20G左右。   

## 训练步骤
首先准备好训练数据集，数据集摆放格式为：
```
- datasets
  - train
    1.jpg
    2.jpg
    3.jpg
    4.jpg
    5.jpg
    .......
  - metadata.jsonl
```
metadata.jsonl中每一行代表一个样本，file_name代表样本的相对路径，text代表样本对应的文本。
```
{"file_name": "train/1000268201_693b08cb0e.jpg", "text": "A child in a pink dress is climbing up a set of stairs in an entry way ."}
```

可首先使用上述提供的Flickr8k数据集为例进行尝试。

## 步骤
1. 下载完库后解压，在百度网盘下载权值，放入model_data
2. 运行GUI.PY


## Reference
https://github.com/lllyasviel/ControlNet   
https://github.com/CompVis/stable-diffusion  
