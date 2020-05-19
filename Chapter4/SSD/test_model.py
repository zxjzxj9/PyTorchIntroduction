from utils import DefaultBoxes, VOCDetection, Encoder, COCODetection
from base_model import VGG16, Loss
from utils import SSDTransformer
from ssd300 import SSD300
from ssd512 import SSD512
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import shutil
import os
import time
import numpy as np
#from apex.parallel import DistributedDataParallel as DDP
#import gc

def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))

# For SSD 300
def test300():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [30, 60, 111, 162, 213, 264, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    print(dboxes().shape)

    img = torch.randn(1, 3, 300, 300)
    model = SSD300(21)
    loc, conf =  model(img)
    print(loc.shape, conf.shape)


def test512():
    figsize = 512
    feat_size = [64, 32, 16, 8, 4, 2, 1]
    steps = [8, 16, 32, 84, 128, 256, 512]
    scales = [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    print(dboxes().shape)

    img = torch.randn(1, 3, 512, 512)
    model = SSD512(21)
    loc, conf =  model(img)
    print(loc.shape, conf.shape)


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    #scales = [30, 60, 111, 162, 213, 264, 315]
    #scales = [21, 45, 101, 157, 213, 269, 325]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes
    
def dboxes512_coco():
    figsize = 512
    feat_size = [64, 32, 16, 8, 4, 2, 1]
    steps = [8, 16, 32, 84, 128, 256, 512]
    # According to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py
    scales = [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def dboxes300():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [30, 60, 111, 162, 213, 264, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def dboxes512():
    figsize = 512
    feat_size = [64, 32, 16, 8, 4, 2, 1]
    steps = [8, 16, 32, 84, 128, 256, 512]
    scales = [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes 

def val300(path):
    ssd300 = SSD300(21)
    dboxes = dboxes300()
    encoder = Encoder(dboxes)
    trans = SSDTransformer(dboxes, (300, 300), val=True)
    valmodel(ssd300, path, dboxes, trans, encoder)

def val512(path):
    ssd512 = SSD512(21)
    dboxes = dboxes512()
    encoder = Encoder(dboxes)
    trans = SSDTransformer(dboxes, (512, 512), val=True)
    valmodel(ssd512, path, dboxes, trans, encoder)

def valmodel(model, path, dboxes, trans, encoder):
    
    od = torch.load(path)
    #print(od.keys())
    model.load_state_dict(od["model"])
    model.eval()
    model.cuda()
    lm = od["label_map"]
    #print(lm)

    img_folder = "../../VOCdevkit/VOC2007/JPEGImages"
    ann_folder = "../../VOCdevkit/VOC2007/Annotations"
    tgt_folder = "../../VOCdevkit/VOC2007/ImageSets/Main/test.txt"
    vd = VOCDetection(img_folder, ann_folder, tgt_folder, label_map=lm, \
                      transform = trans)


    #print(vd.label_map)
    #import sys; sys.exit()

    if not os.path.exists("pr_data"):
        os.mkdir("pr_data")
    else:
        shutil.rmtree("pr_data")
        os.mkdir("pr_data") 


    img_info = [[] for _ in range(21)]

    end = time.time()

    for idx, fname in enumerate(vd.images):
        print("Parse image: {}/{}".format(idx+1, len(vd)), end="\r")
        img, (h, w), bbox, label = vd[idx]
        with torch.no_grad():
            ploc, plabel = model(img.unsqueeze(0).cuda())
     
            try:
                result = encoder.decode_batch(ploc, plabel, 0.50, 200)[0]
            except:
                #raise
                print("No object detected in idx: {}".format(idx), end="\r")
                continue

            loc, label, prob = [r.cpu().numpy() for r in result]
            for loc_, label_, prob_ in zip(loc, label, prob):
                img_info[label_].append((fname[0].split(".")[0], prob_,
                                         loc_[0]*w, loc_[1]*h, \
                                         loc_[2]*w, loc_[3]*h))
    print("")
    print("Test: total time elapsed: {:.3f}".format(time.time() - end))

    for i in range(1, 21):
        fn = "pr_data/pred_"+vd.label_map[i]+".txt"
        with open(fn, "w") as fout:
            for rec in img_info[i]:
                fout.write("{} {:.4f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(*rec))

    from eval import voc_eval
    import glob
    s = 0
    files = glob.glob("./pr_data/pred_*.txt")
    files.sort()

    for f in files:
        name = (f.split("_")[-1]).split(".")[0]
        r = voc_eval(f, annopath=os.path.join(ann_folder, "%s.xml"),
                        imagesetfile=vd.file_filter,
                        classname=name, cachedir="./cache",  ovthresh=0.45)
        s += r[-1]

    s/=20
    print('mAP {:.3f}'.format(s))

    return s


def train300():
    label_map = {}
    dboxes = dboxes300()
    trans = SSDTransformer(dboxes, (300, 300), val=False)
    img_folder = "../../VOCdevkit/VOC2007/JPEGImages"
    ann_folder = "../../VOCdevkit/VOC2007/Annotations"
    tgt_folder = "../../VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"
    vd = VOCDetection(img_folder, ann_folder, tgt_folder, label_map=label_map, \
                      transform = trans)
    dataloader = DataLoader(vd, batch_size=32, shuffle=True, num_workers=8)
    nepochs = 800

    ssd300 = SSD300(21)
    ssd300.train()
    ssd300.cuda()

    loss_func = Loss(dboxes)
    loss_func.cuda()

    optim = torch.optim.SGD(ssd300.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    print("epoch", "nbatch", "loss") 
    

    iter_num = 0
    avg_loss = 0.0
    for epoch in range(nepochs):

        if iter_num >= 60000: 
            break

        for nbatch, (img, img_size, bbox, label) in enumerate(dataloader):
            #gc.collect()
            iter_num += 1

            if iter_num == 40000:
                print("")
                print("lr decay step #1")
                for param_group in optim.param_groups:
                    param_group['lr'] = 1e-4
            if iter_num == 50000:
                print("")
                print("lr decay step #2")
                for param_group in optim.param_groups:
                    param_group['lr'] = 1e-5

            img = Variable(img.cuda(), requires_grad=True)
            ploc, plabel = ssd300(img)
            #torch.cuda.synchronize()
            #show_memusage()

            gloc, glabel = Variable(bbox.transpose(1,2).contiguous().cuda(), 
                               requires_grad=False), \
                           Variable(label.cuda(), requires_grad=False)

            loss = loss_func(ploc, plabel, gloc, glabel)
            #torch.cuda.synchronize()
            #show_memusage()

            avg_loss = 0.999*avg_loss + 0.001*loss.item()
            print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}"\
                        .format(iter_num, loss.item(), avg_loss), end="\r")
    
            optim.zero_grad()
            loss.backward()
            #torch.cuda.synchronize()
            #show_memusage()
            optim.step()

            if iter_num % 5000 == 0:
                print("")
                print("saving model...")
                torch.save({"model" : ssd300.state_dict(), "label_map": vd.label_map}, 
                           "./models/iter_{}.pt".format(iter_num))


def train512():
    label_map = {}
    dboxes = dboxes512()
    trans = SSDTransformer(dboxes, (512, 512), val=False)
    img_folder = "../../VOCdevkit/VOC2007/JPEGImages"
    ann_folder = "../../VOCdevkit/VOC2007/Annotations"
    tgt_folder = "../../VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"
    vd = VOCDetection(img_folder, ann_folder, tgt_folder, label_map=label_map, \
                      transform = trans)

    dataloader = DataLoader(vd, batch_size=32, shuffle=True, num_workers=4)
    nepochs = 800

    ssd512 = SSD512(21)
    ssd512.train()
    ssd512.cuda()

    loss_func = Loss(dboxes)
    loss_func.cuda()

    optim = torch.optim.SGD(ssd512.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    print("epoch", "nbatch", "loss") 

    iter_num = 0
    avg_loss = 0.0
    for epoch in range(nepochs):

        if iter_num >= 60000: 
            break

        for nbatch, (img, img_size, bbox, label) in enumerate(dataloader):
            #gc.collect()
            iter_num += 1

            if iter_num == 40000:
                print("")
                print("lr decay step #1")
                for param_group in optim.param_groups:
                    param_group['lr'] = 1e-4
            if iter_num == 50000:
                print("")
                print("lr decay step #2")
                for param_group in optim.param_groups:
                    param_group['lr'] = 1e-5

            img = Variable(img.cuda(), requires_grad=True)
            ploc, plabel = ssd512(img)
            #torch.cuda.synchronize()
            #show_memusage()

            gloc, glabel = Variable(bbox.transpose(1,2).contiguous().cuda(), 
                               requires_grad=False), \
                           Variable(label.cuda(), requires_grad=False)

            loss = loss_func(ploc, plabel, gloc, glabel)
            #torch.cuda.synchronize()
            #show_memusage()

            avg_loss = 0.999*avg_loss + 0.001*loss.item()
            print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}"\
                        .format(iter_num, loss.item(), avg_loss), end="\r")
    
            optim.zero_grad()
            loss.backward()
            #torch.cuda.synchronize()
            #show_memusage()
            optim.step()
            
            del img, ploc, plabel, gloc, glabel, loss

            if iter_num % 5000 == 0:
                print("")
                print("saving model...")
                torch.save({"model" : ssd512.state_dict(), "label_map": vd.label_map}, 
                           "./models/iter_{}.pt".format(iter_num))

def train300_coco():
    dboxes = dboxes300_coco()
    trans = SSDTransformer(dboxes, (300, 300), val=False)

    #annotate = "../../coco_ssd/instances_valminusminival2014.json"
    #coco_root = "../../coco_data/val2014"
    #annotate = "../../coco_ssd/train.json"
    #coco_root = "../../coco_data/train2014"
    annotate = "../../coco_ssd/instances_train2017.json"
    coco_root = "../../coco_data/train2017"

    coco = COCODetection(coco_root, annotate, trans)
    print("Number of labels: {}".format(coco.labelnum))
    print("Number of images: {}".format(len(coco)))
    #train_sampler = torch.utils.data.distributed.DistributedSampler(coco)
    dataloader = DataLoader(coco, batch_size=32, shuffle=True, num_workers=4)
    #dataloader = DataLoader(coco, batch_size=8, shuffle=True, num_workers=4, sampler=train_sampler, shuffle=(train_sampler is None))

    nepochs = 800
    ssd300 = SSD300(coco.labelnum)
    #ssd300 = DDP(ssd300)
    ssd300.train()
    ssd300.cuda()

    loss_func = Loss(dboxes)
    loss_func.cuda()
    
    optim = torch.optim.SGD(ssd300.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    print("epoch", "nbatch", "loss")

    iter_num = 0
    avg_loss = 0.0

    #od = torch.load("./models/larger_iter_210000.pt")
    #ssd300.load_state_dict(od["model"])
    #iter_num = 210000
    #optim = torch.optim.SGD(ssd300.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)

    for epoch in range(nepochs):
        #train_sampler.set_epoch(epoch)
        if iter_num >= 240000:
            break

        for nbatch, (img, img_size, bbox, label) in enumerate(dataloader):
            iter_num += 1

            if iter_num == 160000:
                print("")
                print("lr decay step #1")
                for param_group in optim.param_groups:
                    param_group['lr'] = 1e-4

            if iter_num == 200000:
                print("")
                print("lr decay step #2")
                for param_group in optim.param_groups:
                    param_group['lr'] = 1e-5
        
            img = Variable(img.cuda(), requires_grad=True)
            ploc, plabel = ssd300(img)

            gloc, glabel = Variable(bbox.transpose(1,2).contiguous().cuda(), 
                               requires_grad=False), \
                           Variable(label.cuda(), requires_grad=False)
            loss = loss_func(ploc, plabel, gloc, glabel)

            if not np.isinf(loss.item()): avg_loss = 0.999*avg_loss + 0.001*loss.item()

            print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}"\
                        .format(iter_num, loss.item(), avg_loss), end="\r")
            optim.zero_grad()
            loss.backward()
            optim.step()

            if iter_num % 5000 == 0:
                print("")
                print("saving model...")
                torch.save({"model" : ssd300.state_dict(), "label_map": coco.label_info}, 
                           "./models/crowd_iter_{}.pt".format(iter_num))

def val300_coco(model_path):
    print("loading model at {}".format(model_path))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    trans = SSDTransformer(dboxes, (300, 300), val=True)

    #annotate = "../../coco_ssd/instances_minival2014.json"
    #coco_root = "../../coco_data/val2014"
    #annotate = "../../coco_ssd/image_info_test-dev2015.json"
    #coco_root = "../../coco_data/test2015"

    annotate = "../../coco_ssd/instances_val2017.json"
    coco_root = "../../coco_data/val2017"

    cocoGt = COCO(annotation_file=annotate)
    coco = COCODetection(coco_root, annotate, trans)

    model = SSD300(coco.labelnum)

    od = torch.load(model_path)
    model.load_state_dict(od["model"])

    model.eval()
    model.cuda()

    ret = []

    inv_map = {v:k for k,v in coco.label_map.items()}
    start = time.time()
    for idx, image_id in enumerate(coco.img_keys):
        img, (htot, wtot), _, _ = coco[idx]
          
        with torch.no_grad():
            print("Parsing image: {}/{}".format(idx+1, len(coco)), end="\r")
            ploc, plabel = model(img.unsqueeze(0).cuda())

            try:
                result = encoder.decode_batch(ploc, plabel, 0.50, 200)[0]
            except:
                #raise
                print("")
                print("No object detected in idx: {}".format(idx), end="\r")
                continue

            loc, label, prob = [r.cpu().numpy() for r in result]
            for loc_, label_, prob_ in zip(loc, label, prob):
                ret.append([image_id, loc_[0]*wtot, \
                                      loc_[1]*htot, 
                                      (loc_[2] - loc_[0])*wtot, 
                                      (loc_[3] - loc_[1])*htot,
                                      prob_,
                                      inv_map[label_]])
    print("")
    print("Predicting Ended, totoal time: {:.2f} s".format(time.time()-start))

    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    #E.params.useSegm = 0
    #E.params.recThrs = [0.5]
    #E.params.maxDets = [10, 100, 200]
    E.evaluate()
    E.accumulate()
    E.summarize()
                                       


if __name__ == "__main__":
    #test300()
    torch.backends.cudnn.benchmark = True
    #test512()
    #train512()
    #val512("./models/iter_60000.pt")
    #val512("./models/iter_5000.pt")
    #val300("./models/iter_60000.pt")
    #train300()
    #train300_coco()
    import sys
    val300_coco(sys.argv[1])
    #val300_coco("models/iter_150000.pt")
