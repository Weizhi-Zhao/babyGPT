# babyGPT

作者：赵为之

感谢：Karpathy-nanoGPT

# 环境配置

```bash
pip install -r requirements.txt
```

## GPTV：图片描述多模态模型

### 数据准备

1. 将存有图片的`Train`、`Val`文件夹放入`data/image_caption/`路径下。
2. 使用InternVL2-8B模型，生成对这些图片的描述，作为ground truth。InternVL2-8B模型可以根据[swift教程](https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/internvl%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.html)安装。
```bash
cd data/image_caption/
python nlp_gen_data.py
```
3. 准备预训练数据，下载[书生·万象数据集](https://opendatalab.com/OpenDataLab/WanJuan1_dot_0)中的jsonl文件（一个压缩包即可），解压到`data/image_caption/pretrain/`路径下，执行`prepare.py`生成预训练数据。
```bash
cd data/image_caption/pretrain
python prepare.py
```
4. 生成训练数据和分词器。
```bash
cd data/image_caption
python prepare.py
```

### 训练

1. 预训练语言模型。
```bash
python python GPTV_pretrain.py --config configs/GPTV_pretrain.yaml --name GPTV_pretrain --save_log
```

2. 训练视觉-语言多模态模型。
```bash
python python train_GPTV.py --config configs/GPTV.yaml --name GPTV --save_log
```

### 测试

1. 生成测试集的图片描述。
```bash
python ???
```

2. 为图片描述打分。
```bash
python ???
```