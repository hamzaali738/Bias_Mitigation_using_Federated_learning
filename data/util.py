# '''Modified from https://github.com/alinlab/LfF/blob/master/data/util.py'''

# import os
# import torch
# from torch.utils.data.dataset import Dataset, Subset
# from torchvision import transforms as T
# from glob import glob
# from PIL import Image
# from data.attr_dataset import AttributeDataset

# class IdxDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return (idx, *self.dataset[idx])


# class ZippedDataset(Dataset):
#     def __init__(self, datasets):
#         super(ZippedDataset, self).__init__()
#         self.dataset_sizes = [len(d) for d in datasets]
#         self.datasets = datasets

#     def __len__(self):
#         return max(self.dataset_sizes)

#     def __getitem__(self, idx):
#         items = []
#         for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
#             items.append(self.datasets[dataset_idx][idx % dataset_size])

#         item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

#         return item

# class CMNISTDataset(Dataset):
#     def __init__(self,root,split,transform=None, image_path_list=None):
#         super(CMNISTDataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.image2pseudo = {}
#         self.image_path_list = image_path_list

#         if split=='train':
#             self.align = glob(os.path.join(root, 'align',"*","*"))
#             self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
#             self.data = self.align + self.conflict
#         elif split=='valid':
#             self.data = glob(os.path.join(root,split,"*"))            
#         elif split=='test':
#             self.data = glob(os.path.join(root, '../test',"*","*"))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)
        
#         return image, attr, self.data[index]


# class WaterbirdsDataset(Dataset):
#     def __init__(self, root, split, transform=None, image_path_list=None):
#         super(WaterbirdsDataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.image_path_list = image_path_list

#         if split == "train":
#             self.align = glob(os.path.join(root, "align", "*", "*"))
#             self.conflict = glob(os.path.join(root, "conflict", "*", "*"))
#             self.data = self.align + self.conflict
#         elif split == "valid":
#             self.data = glob(os.path.join(root, split, "*"))
#         elif split == "test":
#             self.data = glob(os.path.join(root, "../test", "*", "*"))
        
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, index):
#         filename = os.path.basename(self.data[index])
#         parts = filename.split('_')

#         if len(parts) < 3:
#             raise ValueError(f"[DEBUG] Skipping bad file: {self.data[index]}")

#         attr = torch.LongTensor([
#             int(parts[-2]),
#             int(parts[-1].split('.')[0])
#         ])

#         image = Image.open(self.data[index]).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, attr, self.data[index]


# class CIFAR10Dataset(Dataset):
#     def __init__(self, root, split, transform=None, image_path_list=None, use_type0=None, use_type1=None):
#         super(CIFAR10Dataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.image2pseudo = {}
#         self.image_path_list = image_path_list

#         if split=='train':
#             self.align = glob(os.path.join(root, 'align',"*","*"))
#             self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
#             self.data = self.align + self.conflict

#         elif split=='valid':
#             self.data = glob(os.path.join(root,split,"*", "*"))

#         elif split=='test':
#             self.data = glob(os.path.join(root, '../test',"*","*"))
#             data_conflict = []
#             data_align = []
#             for path in self.data:
#                 target_label = path.split('/')[-1].split('.')[0].split('_')[1]
#                 bias_label = path.split('/')[-1].split('.')[0].split('_')[2]

#                 if target_label != bias_label:
#                     data_conflict.append(path)
#                 if target_label == bias_label:
#                     data_align.append(path)
#             self.data_conflict = data_conflict
#             self.data_align = data_align
#             self.data = data_conflict + data_align


# class BmnistDataset(Dataset):
#     def __init__(self, root, split, transform=None, image_path_list=None):
#         super(BmnistDataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.image2pseudo = {}
#         self.image_path_list = image_path_list

#         if split=='train':
#             self.data  = glob(os.path.join(root, split,"*","*"))
#             print(len(self.data))

#         elif split=='valid':
#             self.data = glob(os.path.join(root,split,"*","*"))  
                   
#         elif split=='test':
#             print(os.path.join(root, split))
#             self.data = glob(os.path.join(root, split,"*","*"))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         attr = torch.LongTensor([int(self.data[index].split('/')[-2])])
#         image = Image.open(self.data[index]).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)

#         return image, attr, self.data[index]
# class bFFHQDataset(Dataset):
#     def __init__(self, root, split, transform=None, image_path_list=None):
#         super(bFFHQDataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.image2pseudo = {}
#         self.image_path_list = image_path_list

#         if split=='train':
#             self.align = glob(os.path.join(root, 'align',"*","*"))
#             self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
#             self.data = self.align + self.conflict

#         elif split=='valid':
#             self.data = glob(os.path.join(os.path.dirname(root), split, "*"))

#         elif split=='test':
#             self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
#             data_conflict = []
#             for path in self.data:
#                 target_label = path.split('/')[-1].split('.')[0].split('_')[1]
#                 bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
#                 if target_label != bias_label:
#                     data_conflict.append(path)
#             self.data = glob(os.path.join(root, split,"*","*"))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         attr = torch.LongTensor([int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)  
#         return image, attr, self.data[index]

# class BARDataset(Dataset):
#     def __init__(self, root, split, transform=None, percent=None, image_path_list=None):
#         super(BARDataset, self).__init__()
#         self.transform = transform
#         self.percent = percent
#         self.split = split
#         self.image2pseudo = {}
#         self.image_path_list = image_path_list

#         self.train_align = glob(os.path.join(root,'train/align',"*/*"))
#         self.train_conflict = glob(os.path.join(root,'train/conflict',f"{self.percent}/*/*"))
#         self.valid = glob(os.path.join(root,'valid',"*/*"))
#         self.test = glob(os.path.join(root,'test',"*/*"))

#         if self.split=='train':
#             self.data = self.train_align + self.train_conflict
#         elif self.split=='valid':
#             self.data = self.valid
#         elif self.split=='test':
#             self.data = self.test

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         attr = torch.LongTensor(
#             [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')
#         image_path = self.data[index]

#         if 'bar/train/conflict' in image_path:
#             attr[1] = (attr[0] + 1) % 6
#         elif 'bar/train/align' in image_path:
#             attr[1] = attr[0]

#         if self.transform is not None:
#             image = self.transform(image)  
#         return image, attr, (image_path, index)
    
# class DogCatDataset(Dataset):
#     def __init__(self, root, split, transform=None, image_path_list=None):
#         super(DogCatDataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.image_path_list = image_path_list

#         if split == "train":
#             self.align = glob(os.path.join(root, "align", "*", "*"))
#             self.conflict = glob(os.path.join(root, "conflict", "*", "*"))
#             self.data = self.align + self.conflict
#         elif split == "valid":
#             self.data = glob(os.path.join(root, split, "*"))
#         elif split == "test":
#             self.data = glob(os.path.join(root, "../test", "*", "*"))
        
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         attr = torch.LongTensor(
#             [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)  
#         return image, attr, self.data[index]





# transforms = {
    
#     "bmnist": {
#         "train": T.Compose([T.ToTensor()]), 
#         "valid": T.Compose([T.ToTensor()]), 
#         "test": T.Compose([T.ToTensor()]),
#         },
#     "cmnist": {
#         "train": T.Compose([T.ToTensor()]),
#         "valid": T.Compose([T.ToTensor()]),
#         "test": T.Compose([T.ToTensor()])
#         },
#     "bar": {
#         "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
#         "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
#         "test": T.Compose([T.Resize((224, 224)), T.ToTensor()])
#     },
#     "bffhq": {
#         "train": T.Compose([T.Resize((224,224)), T.ToTensor()]),
#         "valid": T.Compose([T.Resize((224,224)), T.ToTensor()]),
#         "test": T.Compose([T.Resize((224,224)), T.ToTensor()])
#         },
    
#      "wbirds": {
#         "train": T.Compose([T.Resize((256,256)), T.ToTensor()]),
#         "valid": T.Compose([T.Resize((256,256)), T.ToTensor()]),
#         "test": T.Compose([T.Resize((256,256)), T.ToTensor()])
#         },
#     "dogs_and_cats": {
#         "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
#         "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
#         "test": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
#     },
#      "cifar10c": {
#         "train": T.Compose([T.ToTensor(),]),
#         "valid": T.Compose([T.ToTensor(),]),
#         "test": T.Compose([T.ToTensor(),]),
#         },
#     }
    


# transforms_preprcs = {

     
#     "wbirds": {
#         "train": T.Compose([
#             T.Resize(256),
#             T.RandomHorizontalFlip(),
#             T.CenterCrop(224),
#             T.ToTensor(),
#             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

#         ]),
#         "valid": T.Compose([
#             T.Resize(256),
#             T.CenterCrop(224),
#             T.ToTensor(),
#             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

#         ]),
#         "test": T.Compose([
#             T.Resize(256),
#             T.CenterCrop(224),
#             T.ToTensor(),
#             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

#         ]),
#         },

#     "bmnist": {
#         "train": T.Compose([T.ToTensor()]), 
#         "valid": T.Compose([T.ToTensor()]), 
#         "test": T.Compose([T.ToTensor()]),
#         },
#     "cmnist": {
#         "train": T.Compose([T.ToTensor()]),
#         "valid": T.Compose([T.ToTensor()]),
#         "test": T.Compose([T.ToTensor()])
#         },
#     "bar": {
#         "train": T.Compose([
#             T.Resize((224, 224)),
#             T.RandomCrop(224, padding=4),
#             T.RandomHorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ]
#         ),
#         "valid": T.Compose([
#             T.Resize((224, 224)),
#             T.ToTensor(),
#             T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ]
#         ),
#         "test": T.Compose([
#             T.Resize((224, 224)),
#             T.ToTensor(),
#             T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ]
#         )
#     },
#     "bffhq": {
#         "train": T.Compose([
#             T.Resize((224,224)),
#             T.RandomCrop(224, padding=4),
#             T.RandomHorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

#             ]
#         ),
#         "valid": T.Compose([
#             T.Resize((224,224)),
#             T.ToTensor(),
#             T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ]
#         ),
#         "test": T.Compose([
#             T.Resize((224,224)),
#             T.ToTensor(),
#             T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ]
#         )
#         },
#     "dogs_and_cats": {
#             "train": T.Compose(
#                 [
#                     T.Resize((224, 224)),
#                     T.RandomCrop(224, padding=4),
#                     T.RandomHorizontalFlip(),
#                     T.ToTensor(),
#                     T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                 ]
#             ),
#             "valid": T.Compose(
#                 [
#                     T.Resize((224, 224)),
#                     T.ToTensor(),
#                     T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                 ]
#             ),
#             "test": T.Compose(
#                 [
#                     T.Resize((224, 224)),
#                     T.ToTensor(),
#                     T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                 ]
#             ),
#         },
#         "cifar10c": {
#         "train": T.Compose(
#             [
#                 T.RandomCrop(32, padding=4),
#                 # T.RandomResizedCrop(32),
#                 T.RandomHorizontalFlip(),
#                 T.ToTensor(),
#                 T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ]
#         ),
#         "valid": T.Compose(
#             [
#                 T.ToTensor(),
#                 T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ]
#         ),
#         "test": T.Compose(
#             [
#                 T.ToTensor(),
#                 T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ]
#         ),
#     },
# }


# def get_dataset(dataset, data_dir, dataset_split, transform_split, percent=None, use_preprocess=None, image_path_list=None):

#     dataset_category = dataset.split("-")[0]
#     if use_preprocess:
#         transform = transforms_preprcs[dataset_category][transform_split]
#     else:
#         transform = transforms[dataset_category][transform_split]

#     dataset_split = "valid" if (dataset_split == "eval") else dataset_split

#     if dataset == 'cmnist':
#         root = data_dir + f"/cmnist/{percent}"
#         dataset = CMNISTDataset(root=root,split=dataset_split,transform=transform, image_path_list=image_path_list)

#     elif dataset == 'bmnist':
#         root = data_dir + f"/cmnist/{percent}"
#         dataset = BmnistDataset(root=root,split=dataset_split,transform=transform, image_path_list=image_path_list)

#     elif dataset == "bffhq":
#         root = data_dir + f"/bffhq/{percent}"
#         dataset = bFFHQDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)

#     elif dataset == "wbirds":
#         root = data_dir + f'/wbirds/{percent}'
#         dataset = WaterbirdsDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)
        
#     elif dataset == "bar":
#         root = data_dir + f"/bar"
#         dataset = BARDataset(root=root, split=dataset_split, transform=transform, percent=percent, image_path_list=image_path_list)

#     elif dataset == "dogs_and_cats":
#         root = data_dir + f"/dogs_and_cats/{percent}"
#         print(root)
#         dataset = DogCatDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)

#     elif 'cifar10c' in dataset:
#         root = data_dir + f"/cifar10c/{percent}"
#         print(root)
#         dataset = CIFAR10Dataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list, use_type0=None, use_type1=None)


#     # elif 'cifar10c' in dataset:
#     #     # root = data_dir + f"/cifar10c/{percent}"
#     #     # print(root)
#     #     # dataset = CIFAR10Dataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list, use_type0=None, use_type1=None)
#     #     dataset_category = dataset.split("-")[0]
#     #     # root = os.path.join(data_dir, dataset)
#     #     root = "/DATA/divyaansh/download/LFF/debias/CorruptedCIFAR10-Type0-Skewed0.02-Severity1"
#     #     transform = transforms[dataset_category][transform_split]
#     #     dataset_split = "valid" if (dataset_split == "eval") else dataset_split
#     #     dataset = AttributeDataset(
#     #         root=root, split=dataset_split, transform=transform
#     #     )

#     else:
#         print('wrong dataset ...')
#         import sys
#         sys.exit(0)

#     return dataset
'''Modified from https://github.com/alinlab/LfF/blob/master/data/util.py'''

import os
import torch
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms as T
from glob import glob
from PIL import Image
import pandas as pd

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item

class CMNISTDataset(Dataset):
    def __init__(self,root,split,transform=None, image_path_list=None):
        super(CMNISTDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict
        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*"))            
        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image, attr, self.data[index]

class WaterbirdsDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(WaterbirdsDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image_path_list = image_path_list

        if split == "train":
            self.align = glob(os.path.join(root, "align", "*", "*"))
            self.conflict = glob(os.path.join(root, "conflict", "*", "*"))
            self.data = self.align + self.conflict
        elif split == "valid":
            self.data = glob(os.path.join(root, split, "*"))
        elif split == "test":
            self.data = glob(os.path.join(root, "../test", "*", "*"))
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        filename = os.path.basename(self.data[index])
        parts = filename.split('_')

        if len(parts) < 3:
            raise ValueError(f"[DEBUG] Skipping bad file: {self.data[index]}")

        attr = torch.LongTensor([
            int(parts[-2]),
            int(parts[-1].split('.')[0])
        ])

        image = Image.open(self.data[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, attr, self.data[index]




class CIFAR10Dataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None, use_type0=None, use_type1=None):
        super(CIFAR10Dataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))
            data_conflict = []
            data_align = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]

                if target_label != bias_label:
                    data_conflict.append(path)
                if target_label == bias_label:
                    data_align.append(path)
            self.data_conflict = data_conflict
            self.data_align = data_align
            self.data = data_conflict + data_align

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]

class BmnistDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(BmnistDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.data  = glob(os.path.join(root, split,"*","*"))
            print(len(self.data))

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*","*"))  
                   
        elif split=='test':
            print(os.path.join(root, split))
            self.data = glob(os.path.join(root, split,"*","*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('/')[-2])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]
class bFFHQDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))

        # elif split=='test':
        #     self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
        #     data_conflict = []
        #     for path in self.data:
        #         target_label = path.split('/')[-1].split('.')[0].split('_')[1]
        #         bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
        #         if target_label == '1' and target_label == bias_label:
        #             data_conflict.append(path)
        #     self.data = data_conflict

        elif split=='test':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
            data_conflict = []
            data_align = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]

                if target_label != bias_label:
                    data_conflict.append(path)
                if target_label == bias_label:
                    data_align.append(path)
            self.data_conflict = data_conflict
            self.data_align = data_align
            self.data = data_conflict + data_align


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)  
        return image, attr, self.data[index]

class BARDataset(Dataset):
    def __init__(self, root, split, transform=None, percent=None, image_path_list=None):
        super(BARDataset, self).__init__()
        self.transform = transform
        self.percent = percent
        self.split = split
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        self.train_align = glob(os.path.join(root,'train/align',"*/*"))
        self.train_conflict = glob(os.path.join(root,'train/conflict',f"{self.percent}/*/*"))
        self.valid = glob(os.path.join(root,'valid',"*/*"))
        self.test = glob(os.path.join(root,'test',"*/*"))

        if self.split=='train':
            self.data = self.train_align + self.train_conflict
        elif self.split=='valid':
            self.data = self.valid
        elif self.split=='test':
            self.data = self.test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')
        image_path = self.data[index]

        if 'bar/train/conflict' in image_path:
            attr[1] = (attr[0] + 1) % 6
        elif 'bar/train/align' in image_path:
            attr[1] = attr[0]

        if self.transform is not None:
            image = self.transform(image)  
        return image, attr, (image_path, index)
    
class DogCatDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(DogCatDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image_path_list = image_path_list

        if split == "train":
            self.align = glob(os.path.join(root, "align", "*", "*"))
            self.conflict = glob(os.path.join(root, "conflict", "*", "*"))
            self.data = self.align + self.conflict
        elif split == "valid":
            self.data = glob(os.path.join(root, split, "*"))
        elif split == "test":
            self.data = glob(os.path.join(root, "../test", "*", "*"))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)  
        return image, attr, self.data[index]


transforms = {
    
    "bmnist": {
        "train": T.Compose([T.ToTensor()]), 
        "valid": T.Compose([T.ToTensor()]), 
        "test": T.Compose([T.ToTensor()]),
        },
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bar": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor()])
    },
    "bffhq": {
        "train": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224,224)), T.ToTensor()])
        },
    
     "wbirds": {
        "train": T.Compose([T.Resize((256,256)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((256,256)), T.ToTensor()]),
        "test": T.Compose([T.Resize((256,256)), T.ToTensor()])
        },
    "dogs_and_cats": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
    },
     "cifar10c": {
        "train": T.Compose([T.ToTensor(),]),
        "valid": T.Compose([T.ToTensor(),]),
        "test": T.Compose([T.ToTensor(),]),
        },
    }
    


transforms_preprcs = {
    "wbirds": {
        "train": T.Compose([
            T.Resize(256),
            T.RandomHorizontalFlip(),
            T.CenterCrop(224),
            T.ToTensor(),
        ]),
        "valid": T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ]),
        "test": T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ]),
        },

    "bmnist": {
        "train": T.Compose([T.ToTensor()]), 
        "valid": T.Compose([T.ToTensor()]), 
        "test": T.Compose([T.ToTensor()]),
        },
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bar": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        ),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        ),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        )
    },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        },
    "dogs_and_cats": {
            "train": T.Compose(
                [
                    T.Resize((224, 224)),
                    T.RandomCrop(224, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            "valid": T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            "test": T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
        },
        "cifar10c": {
        "train": T.Compose(
            [
                T.RandomCrop(32, padding=4),
                # T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
}


def get_dataset(dataset, data_dir, dataset_split, transform_split, percent, use_preprocess=None, image_path_list=None):

    dataset_category = dataset.split("-")[0]
    if use_preprocess:
        transform = transforms_preprcs[dataset_category][transform_split]
    else:
        transform = transforms[dataset_category][transform_split]

    dataset_split = "valid" if (dataset_split == "eval") else dataset_split

    if dataset == 'cmnist':
        root = data_dir + f"/cmnist/{percent}"
        dataset = CMNISTDataset(root=root,split=dataset_split,transform=transform, image_path_list=image_path_list)

    elif dataset == 'bmnist':
        root = data_dir + f"/cmnist/{percent}"
        dataset = BmnistDataset(root=root,split=dataset_split,transform=transform, image_path_list=image_path_list)

    elif dataset == "bffhq":
        root = data_dir + f"/bffhq/{percent}"
        dataset = bFFHQDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)

    elif dataset == "wbirds":
        root = data_dir + f'/wbirds/5pct'
        dataset = WaterbirdsDataset(root=root, split=dataset_split, transform=transform)
        
    elif dataset == "bar":
        root = data_dir + f"/bar"
        dataset = BARDataset(root=root, split=dataset_split, transform=transform, percent=percent, image_path_list=image_path_list)

    elif dataset == "dogs_and_cats":
        root = data_dir + f"/dogs_and_cats/{percent}"
        print(root)
        dataset = DogCatDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)

    elif 'cifar10c' in dataset:
        root = data_dir + f"/cifar10c/{percent}"
        print(root)
        dataset = CIFAR10Dataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list, use_type0=None, use_type1=None)

    else:
        print('wrong dataset ...')
        import sys
        sys.exit(0)

    return dataset


def create_fl_validation_set(dataset, num_per_class=10):
    """
    Create a balanced validation set from CMNIST training data.
    Selects num_per_class bias-aligned and num_per_class bias-conflicting
    samples for each of 10 classes (total = 10 * 2 * num_per_class = 200).

    Args:
        dataset: CMNISTDataset with .align and .conflict and .data lists
        num_per_class: number of aligned and conflicting samples per class

    Returns:
        val_indices: list of indices into dataset.data for validation
        train_indices: list of remaining indices for training
    """
    import os, random

    # Build per-class aligned / conflicting index lists
    align_by_class = {c: [] for c in range(10)}
    conflict_by_class = {c: [] for c in range(10)}

    for idx, path in enumerate(dataset.data):
        fname = os.path.basename(path)
        parts = fname.replace('.png', '').split('_')
        target_label = int(parts[-2])
        bias_label = int(parts[-1])
        if target_label == bias_label:
            align_by_class[target_label].append(idx)
        else:
            conflict_by_class[target_label].append(idx)

    val_indices = []
    for c in range(10):
        random.shuffle(align_by_class[c])
        random.shuffle(conflict_by_class[c])
        val_indices.extend(align_by_class[c][:num_per_class])
        val_indices.extend(conflict_by_class[c][:num_per_class])

    val_set = set(val_indices)
    train_indices = [i for i in range(len(dataset.data)) if i not in val_set]

    print(f"[FL Data] Validation set: {len(val_indices)} samples, "
          f"Training set: {len(train_indices)} samples")
    return val_indices, train_indices