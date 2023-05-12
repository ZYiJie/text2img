# -*- encoding: utf-8 -*-
'''
@File    :   domain_img_minor.py
@Time    :   2023/04/06 15:17:13
@Author  :   ZhouYijie 
@Function  :    基于领域词库挖掘ZERO数据集中的数据, 数据地址 https://zero.so.com/
@Usage  :    
    vocab_file 格式: 
        word1 freqence1
        word2 freqence2
        ...
'''
import pandas as pd
import jieba_fast as jieba
from multiprocessing import Process, Manager
import random
import os

SOURCE_DIR = "/home/langchao/zyj/wukong_release/"
THREAD_NUM = 20
THRESHOLD = 0.5
LENGTH_RANGE = (2, 35)
VOCAB_PATH = "food.dic"
OUTPUT_FILE = "food.tsv"
LOG_FILE = "food.log"
jieba.load_userdict(VOCAB_PATH)
VOCAB = set()
with open(VOCAB_PATH, "r") as f:
    for line in f:
        arr = line.strip().split()
        if len(arr) == 2:
            VOCAB.add(arr[0])

print(f"vocab size = {len(VOCAB)}")

def zero_dataset_domain_minor(id, filenames, return_dict):
    ret = []
    for file in random.sample(filenames, len(filenames)):
        path = os.path.join(SOURCE_DIR, file)
        cnt1, cnt2 = 0, 0
        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            caption = row.caption
            url = row.url
            if isinstance(caption, str) and len(caption) in range(*LENGTH_RANGE) and isinstance(url, str):
                words = jieba.lcut(caption)
                hit = 0
                for word in words:
                    if word in VOCAB:
                        hit += 1
                if hit / len(words) >= THRESHOLD:
                    ret.append([caption, url])
                    cnt1 += 1
                cnt2 += 1
        with open(LOG_FILE, "a+") as f:
            f.write(f"[Finished] {path}\toutput: {cnt1}/{cnt2}\n")
    return_dict[id] = ret

def multithreading_process(arr, thread_num, func):
    manager = Manager()
    return_dict = manager.dict()
    chunk_size = int(len(arr) / thread_num)+1
    jobs = []
    for i in range(thread_num):
        begin = i*chunk_size
        end = min((i+1)*chunk_size, len(arr))
        p = Process(target=func, args=(i, arr[begin : end], return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    results = []
    for i in range(thread_num):
        results += return_dict[i]
    print(f"ouput size = {len(results)}")
    return random.sample(results, len(results))


if __name__ == "__main__":
    print("loading data ...")
    filenames = os.listdir(SOURCE_DIR)
    print(f"input file num = {len(filenames)}")
    THREAD_NUM = min(THREAD_NUM, len(filenames))
    results = multithreading_process(filenames, THREAD_NUM, zero_dataset_domain_minor)
    results = pd.DataFrame(results, columns=['caption', 'url'])
    print("caption length statistics")
    print(results["caption"].str.len().describe())
    results.to_csv(OUTPUT_FILE, sep="\t", index=False)    