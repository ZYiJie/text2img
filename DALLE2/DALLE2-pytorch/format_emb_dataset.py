from dalle2_pytorch import HuggingfaceAdapter
from torch.utils.data import DataLoader
import webdataset as wds
from tqdm import tqdm
import numpy as np
import torch
import os
from multiprocessing import Process, Manager


meta_data_path = "../data/imgs/"
batch_size = 32
output_dir = "../data/embs/"

def split2sublist(arr, split_num):
    ret = [[] for _ in range(split_num)]
    for idx, val in enumerate(arr):
        ret[idx % split_num].append(val)
    return ret

def inference(files, device, info_dict):
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
        info_dict[filename] = 0
        for keys, texts, imgs in tqdm(dataloader, desc=filename):
            texts = texts.to(device)
            imgs = imgs.to(device)

            text_embs, text_encodings = clip.embed_text(texts)
            img_embs, _ = clip.embed_image(imgs)
            save_embs[0].append(text_embs)
            save_embs[1].append(text_encodings)
            save_embs[2].append(img_embs)
            save_keys.extend(keys)
            info_dict[filename] += len(keys)

        for dirname, embs in zip(save_dirs, save_embs):
            embs = torch.cat(embs).cpu().numpy()
            assert len(save_keys) == embs.shape[0]
            dic = {
                save_keys[i]:embs[i] for i in range(len(save_keys))
            }
            np.save(os.path.join(output_dir, dirname, filename.replace(".tar", ".npy")), dic)


def multi_processing(arr, thread_num, func):
    sub_arrs = split2sublist(arr, thread_num)
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(thread_num):
        p = Process(target=func, args=(sub_arrs[i], f"cuda:{i}", return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    return return_dict


if __name__ == "__main__":
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
    
    return_dict = multi_processing(filenames, device_cnt, inference)
    for k in return_dict.keys():
        print(k, return_dict[k])