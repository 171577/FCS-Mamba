from data.transform import Transforms
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os

class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt
        self.label_rate = opt.label_rate
        self.phase = opt.phase

        self.use_weak_supervision = bool(getattr(opt, 'use_weak_supervision', False))
        self.use_weak_label_for_sim = bool(getattr(opt, 'use_weak_label', False))
        self.use_weak_label_for_main = bool(getattr(opt, 'use_weak_label_for_main', False))
        self.use_mixed_supervision = bool(getattr(opt, 'use_mixed_supervision', False))
        self.need_weak_label = (self.use_weak_label_for_main or self.use_mixed_supervision) or (
            self.use_weak_supervision and self.use_weak_label_for_sim
        )

        file_root = opt.dataroot

        self.img_names = open(os.path.join(file_root, opt.phase, 'list', f'{opt.phase}.txt')).read().splitlines()
        self.t1_paths = [file_root + '/' + opt.phase + '/A/' + x for x in self.img_names]
        self.t2_paths = [file_root + '/' + opt.phase + '/B/' + x for x in self.img_names]
        self.label_paths = [file_root + '/' + opt.phase + '/label/' + x for x in self.img_names]
        self.label_weak_paths = None
        if self.need_weak_label:
            self.label_weak_paths = [file_root + '/' + opt.phase + '/label_weak/' + x for x in self.img_names]

        if self.phase == 'train':
            if opt.label_rate is not None:
                self.label_img_names = open(file_root + '/' + opt.phase + '/list/' + opt.phase + '_semi_' + opt.label_rate +'.txt').read().splitlines()
                self.label_t1_paths = [file_root + '/' + opt.phase + '/A/' + x for x in self.label_img_names]
                self.label_t2_paths = [file_root + '/' + opt.phase + '/B/' + x for x in self.label_img_names]
                self.label_label_paths = [file_root + '/' + opt.phase + '/label/' + x for x in self.label_img_names]

        self.normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        self.transform = Transforms()
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.t1_paths)

    def __getitem__(self, index):

        t1_path = self.t1_paths[index]
        t2_path = self.t2_paths[index]
        label_path = self.label_paths[index]
        label_weak_path = self.label_weak_paths[index] if self.label_weak_paths is not None else None

        name = self.img_names[index]

        if self.phase == 'train':
            if self.label_rate is not None:
                if t1_path in self.label_t1_paths:
                    with_label = True
                else:
                    with_label = False
            else:
                with_label = True
        else:
            with_label = True

        img1 = Image.open(t1_path).convert('RGB')
        img2 = Image.open(t2_path).convert('RGB')
        label = np.array(Image.open(label_path).convert('L')) // 255
        label = Image.fromarray(label)

        label_weak = None
        if label_weak_path is not None:
            label_weak = np.array(Image.open(label_weak_path).convert('L')) // 255
            label_weak = Image.fromarray(label_weak)

        if self.opt.phase == 'train':
            if label_weak is not None:
                data = self.transform({'img1': img1, 'img2': img2, 'label': label, 'label_weak': label_weak})
                img1, img2, label, label_weak = data['img1'], data['img2'], data['label'], data['label_weak']
            else:
                data = self.transform({'img1': img1, 'img2': img2, 'label': label})
                img1, img2, label = data['img1'], data['img2'], data['label']
        target_size = (int(getattr(self.opt, 'input_size', 256)), int(getattr(self.opt, 'input_size', 256)))
        img1 = img1.resize(target_size, resample=Image.BILINEAR)
        img2 = img2.resize(target_size, resample=Image.BILINEAR)
        label = label.resize(target_size, resample=Image.NEAREST)
        if label_weak is not None:
            label_weak = label_weak.resize(target_size, resample=Image.NEAREST)

        img1 = self.normalize(self.to_tensor(img1)).clone()
        img2 = self.normalize(self.to_tensor(img2)).clone()
        label = torch.from_numpy(np.array(label, dtype=np.int64)).clone()

        if label_weak is not None:
            label_weak = torch.from_numpy(np.array(label_weak, dtype=np.int64)).clone()

        input_dict = {'img1': img1, 'img2': img2, 'label': label, 'label_flag': with_label, 'name': name}
        if label_weak is not None:
            input_dict['label_weak'] = label_weak

        return input_dict


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, opt):
        def _collate_fn(batch):
            if len(batch) == 0:
                return {}
            out = {}
            keys = batch[0].keys()
            for k in keys:
                vals = [b[k] for b in batch]
                v0 = vals[0]
                if torch.is_tensor(v0):
                    shapes = [tuple(v.shape) for v in vals]
                    if len(set(shapes)) != 1:
                        names = [b.get('name', '') for b in batch]
                        raise RuntimeError(f"Inconsistent tensor shapes for key '{k}': shapes={shapes}, names={names}")
                    out[k] = torch.stack([v.contiguous() for v in vals], dim=0)
                elif isinstance(v0, (int, float)):
                    out[k] = torch.tensor(vals)
                else:
                    out[k] = vals
            return out

        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=opt.phase=='train',
                                                      pin_memory=True,
                                                      drop_last=opt.phase=='train',
                                                      num_workers=int(opt.num_workers),
                                                      collate_fn=_collate_fn)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
