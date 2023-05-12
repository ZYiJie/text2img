import sys, os
import torch
import torch.utils.data as data
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
from PIL import Image

sys.path.append('/home/yjw/ZYJ_WorkSpace/BriVL-1.0/BriVL-code-inference')
from utils import getLanMask
from models import build_network
from dataset import build_moco_dataset
from utils.config import cfg_from_yaml_file, cfg
from transformers import AutoTokenizer

'''
Usage: python clip_sort.py [text] [img_num] [save_path]
'''
if len(sys.argv) == 4:
    text = sys.argv[1]
    img_num = int(sys.argv[2])
    save_path = sys.argv[3]
else:
    print('Usage: python clip_sort.py [text] [img_num] [save_path]')
    exit(1)


load_checkpoint = '/home/yjw/ZYJ_WorkSpace/BriVL-1.0/BriVL-pretrain-model/BriVL-1.0-5500w.pth'
cfg_from_yaml_file('/home/yjw/ZYJ_WorkSpace/BriVL-1.0/BriVL-code-inference/cfg/test_xyb.yml', cfg)

model = build_network(cfg.MODEL)
model = model.cuda()
model_component = torch.load(load_checkpoint, map_location=torch.device('cpu'))
model.learnable.load_state_dict(model_component['learnable'])    ####### only save learnable
model = torch.nn.DataParallel(model)
model.eval()
print('Load CLIP model from {:s}'.format(load_checkpoint))


# data
class CLIP_DATA(data.Dataset):
    def __init__(self):
        self.imgnames = []
        self.sentences = []
        for i in range(img_num):
            self.imgnames.append(os.path.join(save_path, f'{i}.jpg'))
            self.sentences.append(text)
        # for i in range(10):
        #     self.imgnames.append(f'./gen_test/{i}.jpg')
        #     self.sentences.append('乞讨小女孩好可怜')
        # print(self.imgnames)
        self.cfg = cfg
        self.text_transform = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.visual_transform = T.Compose([
                                    T.ToTensor(),
                                    T.Resize((456, 456))
                                    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    def __len__(self):
        return len(self.imgnames)
    def __getitem__(self, index):
        new_size = 456
        ################################## image 
        # img_path = os.path.join(self.data_dir , self.imgnames[index]) 
        img_path = self.imgnames[index]
        image = Image.open(img_path).convert('RGB')
        
        img_box_s = []
        img_box_s.append(torch.from_numpy(np.array([0, 0, new_size, new_size]).astype(np.float32))) # bbox number:  self.cfg.MODEL.MAX_IMG_LEN
        
        valid_len = len(img_box_s)
        img_len = torch.full((1,), valid_len, dtype=torch.long)

        if valid_len < self.cfg.MODEL.MAX_IMG_LEN:
            for i in range(self.cfg.MODEL.MAX_IMG_LEN - valid_len):
                img_box_s.append(torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)))   

        image_boxs = torch.stack(img_box_s, 0) # <36, box_grid>

        image = self.visual_transform(image)

        ################################## text 
        imgdir_prefix  = self.imgnames[index]
        text = self.sentences[index]
        
        text_info = self.text_transform(text, padding='max_length', truncation=True,
                                        max_length=self.cfg.MODEL.MAX_TEXT_LEN, return_tensors='pt')
        text = text_info.input_ids.reshape(-1)
        text_len = torch.sum(text_info.attention_mask)

        return image, img_len, text, text_len, image_boxs

clip_data = CLIP_DATA()
dataloader = torch.utils.data.DataLoader(
        clip_data,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
np_text, np_img = None, None
with torch.no_grad():
    num_samples = len(dataloader)
    for idx, batch in enumerate(tqdm(dataloader)):

        # data 
        imgs = batch[0]  # <batchsize, 3, image_size, image_size>
        #print(imgs.size())
        img_lens = batch[1].view(-1)
        texts = batch[2]  # <batchsize, 5, max_textLen>
        text_lens = batch[3] # <batchsize, 5, >
        image_boxs = batch[4] # <BSZ, 36, 4>

        bsz, textlen = texts.size(0), texts.size(1)
        # get image mask
        # print(imgs.size(), texts.size())
        imgMask = getLanMask(img_lens, cfg.MODEL.MAX_IMG_LEN)
        imgMask = imgMask.cuda()

        # get language mask
        textMask = getLanMask(text_lens, cfg.MODEL.MAX_TEXT_LEN)
        textMask = textMask.cuda()

        imgs = imgs.cuda()
        texts = texts.cuda()
        image_boxs = image_boxs.cuda() # <BSZ, 36, 4>


        text_lens = text_lens.cuda() ############
        feature_group = model(imgs, texts, imgMask, textMask, text_lens, image_boxs, is_training=False)
        img, text = feature_group['img_text'] # img2text_text / img_text2img 

        if np_img is None:
            np_img = img.cpu().numpy() # <bsz, featdim>
            np_text = text.cpu().numpy() # <bsz, cap_num, featdim>
    
        else:
            np_img = np.concatenate((np_img, img.cpu().numpy()), axis=0)
            np_text = np.concatenate((np_text, text.cpu().numpy()), axis=0)

N = np_text.shape[0]

img = torch.from_numpy(np_img).cuda() # <N, featdim>
text = torch.from_numpy(np_text).cuda() # <N, featdim>

print(img.size(), text.size(), N)
scores = torch.zeros((N, N), dtype=torch.float32).cuda()
print('Pair-to-pair: calculating scores')
for i in tqdm(range(N)): # row: image  col: text
    scores[i, :] = torch.sum(img[i] * text, -1)
# print(scores)
# ground truth 
recall_k_s = [1, 5, 10] # [1, 5, 10]
GT_label = torch.arange(0, N).view(N, 1).cuda()
# text2img
logits = scores.T
indices = torch.argsort(logits, descending=True) # dim=-1  <N, N>
print('#'*10, indices[0].cpu().numpy())

