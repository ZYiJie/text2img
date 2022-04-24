# text2img

### 项目描述

模式识别课设代码：

- 基于Wukong中文开源数据集，进行CLIP图文检索模型、VQVAE图像编解码模型训练，VQVAE+VQGAN、VQGAN+DALLE图像生成模型训练，开源中文图文检索模型BriVL与自训CLIP性能比较。
- 最后使用VQGAN的进行DALLE模型训练，实现了食物领域的文本生成图像，并使用BriVL重排生成结果。

### 环境依赖

- 主要基于pytorch环境
- 具体环境依赖参考各个项目下的README

### 目录结构 & 重要代码文件

```
│  README.md
│  dalle_gen.py  调用训练得到的DALLE模型，指定文本生成对应图像
│  clip_sort.py  调用BriVL对生成结果进行rerank
├─form_data
│  ├─dealData_wukong.ipynb  对wukong原始数据的下载、清洗以及划分train&val
│  └─foodNames.txt  食物关键词，用于从wukong数据中筛选食物相关的图文对
├─open_clip   CLIP模型训练项目
├─BriVL-code-inference  BriVL模型推理代码
│  
└─DALLE-pytorch  VQVAE模型训练 DALLE模型训练

```

### 运行

- 训练CLIP模型

  ```
  cd open_clip
  nohup python  -u  -m torch.distributed.launch --nproc_per_node=4  src/training/main.py \
    --save-frequency 10 \
    --report-to wandb \
    --train-data="../train_clean.csv"  \  
    --val-data="../val_clean.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --batch-size=64 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=30 \
    --workers=4 \
    --horovod  \
    --model RN50 >log 2>&1 &
  ```

- 训练VQVAE模型
  
  ```
  cd DALLE-pytorch
  nohup python -u train_vae.py \
    --image_folder=训练所用存放图像文件的目录 \
    --batch_size=64 \
    --num_layers=4 \
    --num_tokens=1024 \
    --emb_dim=256 \
    --image_size=256 >vae.log 2>&1 &
  ```

- 训练DALLE模型

  ```
  cd DALLE-pytorch
  nohup python -u train_dalle.py
    --image_text_folder=./DALLE.train.jsonl
    --vae_path= 训练得到的VQVAE模型位置 
    --chinese 
    --amp 
    --dalle_output_file_name=vae_dalle >dalle.log 2>&1 &
  ```

- 使用BriVL进行推理

  ```
  更新BriVL-code-inference/cfg/test_xyb.yml中的JSONPATH
  cd BriVL-code-inference/evaluation
  sh evaluation.sh
  ```

- 使用DALLE生成图像&BriVL推理

  ```
  python dalle_gen.py [text] [img_num] [save_path]
    基于句子生成图像

  python clip_sort.py [text] [img_num] [save_path]
    基于句子和图像计算相似度，输出排序后结果
  
  parameter：
    text：输入食物相关的句子，长度最好大于10
    img_num：生成图像数量
    save_path：图像保存地址
  ```

### 相关资源

- wukong数据集：https://wukong-dataset.github.io/wukong-dataset/benchmark.html
- 使用的开源预训练VQGAN：https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1
- 中文文本tokenizer：https://huggingface.co/hfl/chinese-roberta-wwm-ext

### 参考项目

- https://github.com/mlfoundations/open_clip
- https://github.com/BAAI-WuDao/BriVL
- https://github.com/lucidrains/DALLE-pytorch