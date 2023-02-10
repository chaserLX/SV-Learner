from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import os
import json
import torch
from autoaugment import CIFAR10Policy, ImageNetPolicy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class=50):
        self.root = root_dir + 'imagenet/val/'
        self.transform = transform
        self.val_data = []
        with open(os.path.join('/data/liangxin/webvision/', 'info/synsets.txt')) as f:
            lines = f.readlines()
        synsets = [x.split()[0] for x in lines]
        for c in range(num_class):
            class_path = os.path.join(self.root, synsets[c])
            imgs = os.listdir(self.root + synsets[c])
            for img in imgs:
                self.val_data.append([c, os.path.join(class_path, img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)


class webvision_dataset(Dataset):
    def __init__(
        self, sample_ratio,
        root,
        transform,
        mode,
        pred=[],
        probability=[],
        num_class=50,
        class_flex_adjust=[],
        gmm_prob=[],
    ):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_samples = 0
        self.sample_ratio = sample_ratio

        if self.mode == 'test':
            with open(self.root + 'info/val_filelist.txt') as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
        else:

            with open(self.root + 'info/train_filelist_google.txt') as f:
                lines = f.readlines()
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    train_imgs.append(img)
                    self.train_labels[img] = target
            self.num_samples = len(self.train_labels)


            save_file = 'pred_idx_webvision_aug.npz'
            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:
                if self.mode == "labeled":
                    class_ind = {}
                    ## Get the class indices
                    for kk in range(num_class):
                        class_ind[kk] = [i for i, x in enumerate(train_imgs) if self.train_labels[x] == kk]

                    samples = 0
                    for i in range(num_class):
                        samples += int(self.sample_ratio / class_flex_adjust[i] * len(class_ind[i]))

                    pred_idx_gmm = np.zeros(int(samples))
                    pred_idx_js = np.zeros(int(samples))
                    size_pred = 0

                    ## Creating the Class Balance
                    for i in range(num_class):
                        class_indices = np.array(class_ind[i])  ##  Class indices
                        # JS Selection
                        prob_js = np.argsort(probability[class_ind[i]])  ##  Sorted indices for each class
                        # GMM Selection
                        prob_gmm = np.argsort(-1 * gmm_prob[class_indices])
                        size1 = len(class_indices)
                        class_len = int(self.sample_ratio / class_flex_adjust[i] * len(class_indices))

                        try:
                            pred_idx_js[size_pred:size_pred + class_len] = np.array(class_indices)[
                                prob_js[0:class_len].cpu().numpy()].squeeze()
                            pred_idx_gmm[size_pred:size_pred + class_len] = np.array(class_indices)[
                                prob_gmm[0:class_len]].squeeze()
                            size_pred += class_len
                        except:
                            pred_idx_gmm[size_pred:size_pred + size1] = np.array(class_indices)
                            pred_idx_js[size_pred:size_pred + size1] = np.array(class_indices)
                            size_pred += size1

                    pred_idx_gmm = [int(x) for x in list(pred_idx_gmm)]
                    pred_idx_js = [int(x) for x in list(pred_idx_js)]

                    # pred_idx = [i for i in pred_idx_gmm if i in pred_idx_js]
                    pred_idx = pred_idx_gmm + pred_idx_js
                    pred_idx = list(set(pred_idx))
                    np.savez(save_file, index=pred_idx)

                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    js_probability = probability.clone()
                    js_probability[js_probability < 0.5] = 0  ## Weight Adjustment
                    self.probability = [1 - js_probability[i] for i in pred_idx]
                    print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))

                elif self.mode == "unlabeled":
                    pred_idx1 = np.load(save_file)['index']
                    idx = list(range(self.num_samples))
                    pred_idx = [x for x in idx if x not in pred_idx1]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))

    def __getitem__(self, index):
        if self.mode == "labeled":
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(os.path.join(self.root + img_path)).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            
            return img1, img2, img3, img4, target, prob, index

        elif self.mode == "unlabeled":
            img_path = self.train_imgs[index]
            image = Image.open(os.path.join(self.root + img_path)).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4, index
            
        elif self.mode == "all":
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(os.path.join(self.root + img_path)).convert("RGB")
            img = self.transform(image)
            return img, target, index

        elif self.mode == "test":
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(self.root + 'val_images_256/' + img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == "test":
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class webvision_dataloader:
    def __init__(
        self,
        root,
        imagenet_root,
        batch_size,
        num_workers):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.imagenet_root = imagenet_root

        webvision_weak_transform = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.RandomCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        webvision_strong_transform = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.RandomCrop(299),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ]
        )

        self.transform_imagenet = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.transforms = {
            "warmup": webvision_weak_transform,
            "unlabeled": [
                        webvision_weak_transform,
                        webvision_weak_transform,
                        webvision_strong_transform,
                        webvision_strong_transform
                    ],
            "labeled": [
                        webvision_weak_transform,
                        webvision_weak_transform,
                        webvision_strong_transform,
                        webvision_strong_transform
                    ]
        }
        self.transforms_test = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def run(self, sample_r, mode,  pred=[], prob=[], paths=[], class_flex_adjust=[], gmm_prob=[]):
        if mode == "warmup":
            warmup_dataset = webvision_dataset(sample_r,
                self.root,
                transform=self.transforms["warmup"],
                mode="all",
            )

            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return warmup_loader
        elif mode == "train":
            labeled_dataset = webvision_dataset( sample_r,
                self.root, 
                transform=self.transforms["labeled"],
                mode="labeled",
                pred=pred,
                probability=prob,
                class_flex_adjust=class_flex_adjust,
                gmm_prob=gmm_prob
            )
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True, 
                num_workers=self.num_workers, drop_last = True, pin_memory=True
            )
            unlabeled_dataset = webvision_dataset(sample_r,
                self.root, 
                transform=self.transforms["unlabeled"],
                mode="unlabeled",
                pred=pred,
                probability=prob,
            )
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True, 
                num_workers=self.num_workers, drop_last = True, pin_memory=True
            )
            return labeled_loader, unlabeled_loader

        elif mode == "eval_train":
            eval_dataset = webvision_dataset( sample_r,
                self.root, 
                transform=self.transforms_test,
                mode="all",

            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size*10,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return eval_loader

        elif mode == "test":
            test_dataset = webvision_dataset(
                sample_r, self.root,  transform=self.transforms_test, mode="test"
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size*10,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return test_loader

        elif mode == "imagenet":
            imagenet_val = imagenet_dataset(root_dir=self.imagenet_root,
                            transform=self.transform_imagenet)
            imagenet_loader = DataLoader(
                dataset=imagenet_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return imagenet_loader
