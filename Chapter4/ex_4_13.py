""" 本代码为示例代码，演示如何载入ImageNet的图像数据集
"""

def main_worker(gpu, ngpus_per_node, args):
    # ...
    # 设置数据集目录
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # 设置预处理方法
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 训练数据集
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, 
        sampler=train_sampler)

    # 测试数据集
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # ...
