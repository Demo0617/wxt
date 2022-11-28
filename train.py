#训练、测试划分
import clip
import torch
import numpy as np
from model import MyClip
from prepare_data import MyDataset#, collate_fn
import torch.utils.data as data_utils
from itertools import chain


def train(model, data_dir, batch_size, optimizer, criterion, preprocess, n_epoch, device):
    model.train()
    print('reading dataset...')
    flickr = MyDataset(data_dir, preprocess)
    print('building dataloader...')
    data_loader = data_utils.DataLoader(dataset=flickr, batch_size=batch_size, shuffle=True)#, collate_fn=collate_fn)

    for i in range(n_epoch):
        total, total_wrong, j = 0.0, 0.0, 0
        for caps, imgs, names in data_loader:
            j += 1
            ori_img_f, pred_img_f, pred_label = model(imgs, caps)
            loss = criterion(pred_img_f, ori_img_f)
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_total = len(pred_label)
            names = np.array(names)
            pred_names = np.array(names[np.array(pred_label)])
            cur_wrong = ((names == pred_names)==False).sum()
            total += cur_total
            total_wrong += cur_wrong
            acc = (cur_total*1.0 - cur_wrong) / cur_total
            if j % 10 == 0:
                print('epoch: {}, iter: {}, loss: {}, cur_total: {}, cur_wrong: {}'.format(i, j, loss, cur_total, cur_wrong))
        acc = (total*1.0 - total_wrong) / total
        with open('./log.txt', 'w') as f:
            f.write('------------epoch: {}, acc: {}------------'.format(i, acc))

if __name__ == '__main__':
    data_dir = '/raid/xwang/multimodal/flickr8k'
    batch_size = 256
    device = torch.device('cuda:5')
    n_epoch = 500

    print('getting clip')
    clip.available_models()
    clip_model, preprocess = clip.load("ViT-B/32")