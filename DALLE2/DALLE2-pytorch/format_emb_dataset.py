from dalle2_pytorch import HuggingfaceAdapter
from torch.utils.data import DataLoader
import webdataset as wds
from tqdm import tqdm
import numpy as np
import torch
import os
import sys


meta_data_path = "../data/imgs/"
batch_size = 32
output_dir = "../data/embs/"

def split2sublist(arr, split_num):
    ret = [[] for _ in range(split_num)]
    for idx, val in enumerate(arr):
        ret[idx % split_num].append(val)
    return ret

def inference(files, device):
    clip = HuggingfaceAdapter(
            text_encoder_name = '/data/Data/ye/zyj/PTMs/Taiyi-CLIP-Roberta-large-326M-Chinese',
            img_encoder_name = '/data/Data/ye/zyj/PTMs/clip-vit-large-patch14'
        ).to(device)

    def my_collate(batch):
        keys, txts, jpgs = batch
        tokenized_text = clip.tokenizer(
            txts, max_length=clip.max_text_len, 
            padding='max_length', truncation =True, return_tensors="pt"
        )
        jpg_tensor = clip.processor(images=jpgs, return_tensors="pt")["pixel_values"]
        return [keys, tokenized_text, jpg_tensor]

    for filename in files:
        dataset = wds.WebDataset(os.path.join(meta_data_path, filename)).decode("pil").to_tuple("__key__ txt jpg").batched(batch_size)
        dataloader = DataLoader(dataset, collate_fn=my_collate, batch_size=None, num_workers=8)
        save_embs = [[], [], []]
        save_keys = []
        save_dirs = ["text_embs", "text_encodings", "img_embs"]
        cnt = 0
        for keys, texts, imgs in tqdm(dataloader, desc=filename):
            texts = texts.to(device)
            imgs = imgs.to(device)

            text_embs, text_encodings = clip.embed_text(texts)
            img_embs, _ = clip.embed_image(imgs)
            save_embs[0].append(text_embs)
            save_embs[1].append(text_encodings)
            save_embs[2].append(img_embs)
            save_keys.extend(keys)
            cnt += len(keys)

        for dirname, embs in zip(save_dirs, save_embs):
            embs = torch.cat(embs).cpu().numpy()
            assert len(save_keys) == embs.shape[0]
            dic = {
                save_keys[i]:embs[i] for i in range(len(save_keys))
            }
            np.save(os.path.join(output_dir, dirname, filename.replace(".tar", ".npy")), dic)
        print(f"[Finished] {filename} cnt = {cnt}")

if __name__ == "__main__":
    if len(sys.argv) == 1: # 主进程
        # 创建output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            for dirname in ["text_embs", "text_encodings", "img_embs"]:
                _path = os.path.join(output_dir, dirname)
                if not os.path.exists(_path):
                    os.mkdir(_path)
        # input files
        filenames = []
        for filename in os.listdir(meta_data_path):
            if filename.endswith(".tar"):
                filenames.append(filename)
        print(f"files num = {len(filenames)}")

        device_cnt = torch.cuda.device_count()    
        print(f"Available devices = {device_cnt}")

        device_cnt = min(device_cnt, len(filenames))
        sub_arrs = split2sublist(filenames, device_cnt)
        for i in range(device_cnt):
            os.system(f"python format_emb_dataset.py {','.join(sub_arrs[i])} cuda:{i} > logs/cuda:{i}.log 2>&1 &")
    
    elif len(sys.argv) == 3: # 推理进程
        files = sys.argv[1].split(",")
        device = sys.argv[2]
        print(f"[Begin] {device} {files}")
        inference(files, device)

    else:
        print("params error!")