""" 该代码仅为演示类的构造方法所用，并不能实际运行
"""

class VisionDataset(data.Dataset):
    def __init__(self, root, transforms=None, transform=None, 
        target_transform=None):
        # ...

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

class DatasetFolder(VisionDataset):
    def __init__(self, root, loader, extensions=None, transform=None,
        target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root, transform=transform,
            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        self.loader = loader
        # ...

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)
