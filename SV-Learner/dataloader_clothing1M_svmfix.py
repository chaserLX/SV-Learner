from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
from autoaugment import CIFAR10Policy, ImageNetPolicy


class clothing_dataset(Dataset):
    def __init__(
        self, sample_ratio,
        root,
        transform,
        mode,
        num_samples=0,
        pred=[],
        probability=[],
        paths=[],
        num_class=14,
        class_flex_adjust=[],
        gmm_prob=[],
    ):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels  = {}
        self.val_labels   = {}
        self.sample_ratio = sample_ratio

        with open("%s/noisy_label_kv.txt" % self.root, "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = "%s/" % self.root + 'images/' + entry[0][7:]
                self.train_labels[img_path] = int(entry[1])

        with open("%s/clean_label_kv.txt" % self.root, "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = "%s/" % self.root + 'images/' + entry[0][7:]
                self.test_labels[img_path] = int(entry[1])
        
        save_file = 'pred_idx_clothing1M_aug.npz'
        if mode == 'all':
            train_imgs = []
            with open("%s/noisy_train_key_list.txt" % self.root, "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = "%s/" % self.root + 'images/' + l[7:]
                    train_imgs.append(img_path)
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath]
                if (
                    class_num[label] < (num_samples / 14)
                    and len(self.train_imgs) < num_samples
                ):
                    self.train_imgs.append(impath)
                    class_num[label] += 1
            random.shuffle(self.train_imgs)
        
        elif mode == 'labeled':
            train_imgs = paths
            class_ind  = {}
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
                prob_js = np.argsort(probability[class_ind[i]])      ##  Sorted indices for each class
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

            self.train_imgs  = [train_imgs[i] for i in pred_idx]
            js_probability = probability.clone()
            js_probability[js_probability<0.5] = 0                        ## Weight Adjustment 
            self.probability = [1-js_probability[i] for i in pred_idx]
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))

        elif self.mode == "unlabeled":  
            train_imgs = paths 
            pred_idx1 = np.load(save_file)['index']
            idx = list(range(num_samples))
            pred_idx = [x for x in idx if x not in pred_idx1] 
            self.train_imgs = [train_imgs[i] for i in pred_idx]                         
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))

        elif mode == "test":
            self.test_imgs = []
            with open("%s/clean_test_key_list.txt" % self.root, "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = "%s/" % self.root + 'images/' + l[7:]
                    self.test_imgs.append(img_path)
        elif mode == "val":
            self.val_imgs = []
            with open("%s/clean_val_key_list.txt" % self.root, "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = "%s/" % self.root + 'images/' + l[7:]
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.mode == "labeled":
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            
            return img1, img2, img3, img4, target, prob, index

        elif self.mode == "unlabeled":
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4, index
            
        elif self.mode == "all":
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target, img_path

        elif self.mode == "test":
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target
            
        elif self.mode == "val":
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == "test":
            return len(self.test_imgs)
        if self.mode == "val":
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class clothing_dataloader:
    def __init__(
        self,
        root,
        batch_size,
        warmup_batch_size,
        num_batches,
        num_workers    ):
        self.batch_size = batch_size
        self.warmup_batch_size = warmup_batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root

        clothing1m_weak_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ]
        )


        clothing1m_strong_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ]
        )

        self.transforms = {
            "warmup": clothing1m_weak_transform,
            "unlabeled": [
                        clothing1m_weak_transform,
                        clothing1m_weak_transform,
                        clothing1m_strong_transform,
                        clothing1m_strong_transform
                    ],
            "labeled": [
                        clothing1m_weak_transform,
                        clothing1m_weak_transform,
                        clothing1m_strong_transform,
                        clothing1m_strong_transform
                    ]
        }
        self.transforms_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
                ),
            ]
        )

    def run(self, sample_r, mode,  pred=[], prob=[], paths=[], class_flex_adjust=[], gmm_prob=[]):
        if mode == "warmup":
            warmup_dataset = clothing_dataset(sample_r,
                self.root,
                transform=self.transforms["warmup"],
                mode="all",
                num_samples=self.num_batches * self.warmup_batch_size,
            )
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.warmup_batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return warmup_loader
        elif mode == "train":
            labeled_dataset = clothing_dataset( sample_r,
                self.root, 
                transform=self.transforms["labeled"],
                mode="labeled", 
                num_samples=self.num_batches * self.batch_size,
                pred=pred,
                probability=prob,
                paths=paths,
                class_flex_adjust=class_flex_adjust,
                gmm_prob=gmm_prob
            )
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True, 
                num_workers=self.num_workers, drop_last = True
            )
            unlabeled_dataset = clothing_dataset(sample_r,
                self.root, 
                transform = self.transforms["unlabeled"],
                mode = "unlabeled",
                num_samples = self.num_batches * self.batch_size,
                pred = pred,
                probability=prob,
                paths=paths
            )
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True, 
                num_workers=self.num_workers, drop_last = True
            )
            return labeled_loader, unlabeled_loader

        elif mode == "eval_train":
            eval_dataset = clothing_dataset( sample_r,
                self.root, 
                transform=self.transforms_test,
                mode="all",
                num_samples=self.num_batches * self.batch_size,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size*4,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader

        elif mode == "test":
            test_dataset = clothing_dataset(
                sample_r,self.root,  transform=self.transforms_test, mode="test"
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader

        elif mode == "val":
            val_dataset = clothing_dataset(
                sample_r, self.root ,transform=self.transforms_test, mode="val"
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return val_loader
