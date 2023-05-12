from dalle2_pytorch import HuggingfaceAdapter
from torch.utils.data import DataLoader
from accelerate import Accelerator
import webdataset as wds
from tqdm import tqdm
import numpy as np
import torch
import os


meta_data_path = "/home/zyj/VLP/text2img/DALLE2/data/imgs/"
batch_size = 32
output_dir = "/home/zyj/VLP/text2img/DALLE2/data/embs/"
device = "cuda:1"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    for dirname in ["text_embs", "text_encodings", "img_embs"]:
        _path = os.path.join(output_dir, dirname)
        if not os.path.exists(_path):
            os.mkdir(_path)


clip = HuggingfaceAdapter().to(device)
max_text_len = clip.max_text_len
def my_collate(batch):
    keys, txts, jpgs = batch
    tokenized_text = clip.tokenizer(
        txts, max_length=clip.max_text_len, 
        padding='max_length', truncation =True, return_tensors="pt"
    )
    jpg_tensor = clip.processor(images=jpgs, return_tensors="pt")["pixel_values"]
    return [keys, tokenized_text, jpg_tensor]


filenames = []
for filename in os.listdir(meta_data_path):
    if filename.endswith(".tar"):
        filenames.append(filename)
print(f"files num = {len(filenames)}")

for filename in filenames:
    dataset = wds.WebDataset(os.path.join(meta_data_path, filename)).decode("pil").to_tuple("__key__ txt jpg").batched(batch_size)
    dataloader = DataLoader(dataset, collate_fn=my_collate, batch_size=None, num_workers=8)
    save_embs = [[], [], []]
    save_keys = []
    save_dirs = ["text_embs", "text_encodings", "img_embs"]
    for keys, texts, imgs in tqdm(dataloader, desc=filename):
        texts = texts.to(device)
        imgs = imgs.to(device)

        text_embs, text_encodings = clip.embed_text(texts)
        img_embs, _ = clip.embed_image(imgs)
        save_embs[0].append(text_embs)
        save_embs[1].append(text_encodings)
        save_embs[2].append(img_embs)
        save_keys.extend(keys)
    for dirname, embs in zip(save_dirs, save_embs):
        embs = torch.cat(embs).cpu().numpy()
        assert len(save_keys) == embs.shape[0]
        dic = {
            save_keys[i]:embs[i] for i in range(len(save_keys))
        }
        np.save(os.path.join(output_dir, dirname, filename.replace(".tar", ".npy")), dic)

