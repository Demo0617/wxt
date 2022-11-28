import torch
import torch.nn as nn
import numpy as np
import clip


class MyClip(nn.Module):
    """
    define mian_model based on CLIP + cos_similarity
    work_flow: images - text - images
    """

    def __init__(self, clip_model, device):
        super().__init__()
        self.device = device
        self.clip_model = clip_model
        self.cap_linear = nn.Linear(in_features=512, out_features=512, bias=True, requires_grad=True)
        self.img_linear = nn.Linear(in_features=512, out_features=512, bias=True, requires_grad=True)

    def cap2img(self, names, top_labels, gt_caps, image_features, caps_features):
        n = len(top_labels)
        shuffle_caps_feature = []
        for i in range(n):
            shuffle_caps_feature.extend([caps_features[ind] for ind in top_labels[i].numpy()])

        shuffle_caps_feature = torch.stack(shuffle_caps_feature).to(self.device)

        imgs_probs = (100 * shuffle_caps_feature @ image_features.T).softmax(dim=-1)
        top_probs, top_imgs = imgs_probs.cpu().topk(1, dim=-1)
        img_pairs = []
        hard_label = []
        for i in range(5 * n):
            ans = torch.tensor([i // 5, ])
            if top_imgs[i][0] == ans:
                hard_label.append(1)
            else:
                hard_label.append(0)
            i2i_sim = image_features[ans] @ image_features[top_imgs[i].item()]
            img_pairs.append([names[ans], names[top_imgs[i].item()], image_features[ans], image_features[top_imgs[i].item()], i2i_sim.item(), gt_caps[i]])
        return img_pairs, hard_label

    def t2i(self, top_labels, image_features, caps_features):
        #1. top_labels: 1*n - n
        #2. caps ordered by top_labels
        #3. use caps to pre dict img
        #4. return pair[ori_img, pred_img]
        ttop_labels = np.array(top_labels).flatten()
        caps_features = caps_features[top_labels]
        #print('ttop_labels:',ttop_labels)
        t2i_prob = (100.0 * caps_features @ image_features.T).softmax(dim=-1)
        top_probs, itop_labels = t2i_prob.cpu().topk(1, dim=-1)
        itop_labels = np.array(itop_labels).flatten()
        #print('itop_labels:',itop_labels)
        return image_features, image_features[top_labels], itop_labels



    def forward(self, imgs, caps):

        caps_tokens = clip.tokenize(caps).reshape(-1, 77).to(self.device)
        imgs = torch.tensor(np.stack(imgs)).to(self.device)
        with torch.no_grad():
            caps_features = self.clip_model.encode_text(caps_tokens).float()
            image_features = self.clip_model.encode_image(imgs).float()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        caps_features = caps_features / caps_features.norm(dim=-1, keepdim=True)
        caps_features = self.cap_linear(caps_features)
        image_features = self.img_linear(image_features)
        caps_probs = (100.0 * image_features @ caps_features.T).softmax(dim=-1)
        top_probs, top_labels = caps_probs.cpu().topk(1, dim=-1)

        ori_img_f, pre_img_f, pred_label = self.t2i(top_labels, image_features, caps_features)

        return ori_img_f, pre_img_f, pred_label


                                                                                        1,1           Top