# Original code: https://github.com/JiaRenChang/FaceCycle/blob/master/train_expression.py

# It has been modified to include flexibility of training approaches with:
#     - PSM, PSM with curriculum temporal learning, GM with curriculum temporal learning, and PSM Transfer Learning
#     - Possibility to modify loss function weights using interpolation
#     - A couple of other files we save: losses over training epochs, arguments used to train the model
#     - .yaml file to specify all the different training options in a simple and effective way


import os
import sys
# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

# Change the current working directory
os.chdir('../')
print("Current working directory: {0}".format(os.getcwd()))

sys.path.append(os.getcwd()) # Make sure sys path is on FaceCycle_w_Modif


import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import argparse
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from vgg19 import *
import random
from dataloader import Voxall as DA
from models import *
from Loss import *
import torch.utils.data as data
from dataloader import *
import datetime
import json
import math
import pandas as pd
import inspect
import numpy as np
import itertools
import warnings
warnings.filterwarnings('ignore')
import yaml


############################################## PREDEFINED FUNCTIONS ##############################################
# Copied from Chang et al. 2021()
def fast_collate(batch):
    imgs0 = [img[0] for img in batch]
    imgs1 = [img[1] for img in batch]

    w = imgs0[0].size[0]
    h = imgs0[0].size[1]

    tensor0 = torch.zeros( (len(imgs0), 3, h, w), dtype=torch.uint8)
    tensor1 = torch.zeros( (len(imgs1), 3, h, w), dtype=torch.uint8)

    for i, img in enumerate(imgs0):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor0[i] += torch.from_numpy(nump_array)

    for i, img in enumerate(imgs1):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor1[i] += torch.from_numpy(nump_array)

    return tensor0, tensor1

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input0, self.next_input1 = next(self.loader)
        except StopIteration:
            self.next_input0, self.next_input1 = None, None
            return

        with torch.cuda.stream(self.stream):
            self.next_input0 = self.next_input0.cuda(non_blocking=True)
            self.next_input1 = self.next_input1.cuda(non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input0 = self.next_input0.float()
            self.next_input0 = self.next_input0.sub_(self.mean).div_(self.std)
            self.next_input1 = self.next_input1.float()
            self.next_input1 = self.next_input1.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input0 = self.next_input0
        input1 = self.next_input1
        self.preload()
        return input0, input1

def denorm(x):
    x[:,0,:,:] = x[:,0,:,:]*0.229 + 0.485
    x[:,1,:,:] = x[:,1,:,:]*0.224 + 0.456
    x[:,2,:,:] = x[:,2,:,:]*0.225 + 0.406
    return x

def denorm_reto(x):
    y = x.clone()
    y[:,0,:,:] = ((x[:,2,:,:]*0.229 + 0.485)*255-91.4953)
    y[:,1,:,:] = ((x[:,1,:,:]*0.224 + 0.456)*255-103.8827)
    y[:,2,:,:] = ((x[:,0,:,:]*0.225 + 0.406)*255-131.0912)
    return y.contiguous()

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    Bf,Cf,Hf,Wf = flo.size()        
    ## down sample flow
    #scale = H/Hf
    flo = F.upsample(flo, size = (H,W),mode='bilinear', align_corners = True)  # resize flow to x
    flo = torch.clamp(flo,-1,1)
    #flo = flo*scale # rescale flow depend on its size
    ##    
    # mesh grid 
    xs = np.linspace(-1,1,W)
    xs = np.meshgrid(xs, xs)
    xs = np.stack(xs, 2)
    xs = torch.Tensor(xs).unsqueeze(0).repeat(B, 1,1,1).to(device)

    vgrid = Variable(xs, requires_grad=False) + flo.permute(0,2,3,1)         
    output = nn.functional.grid_sample(x, vgrid, align_corners = True)
    
    return output

# Modified 
def adjust_learning_rate(optimizer, epoch,initial_lrate=8e-5,min_lr_to_tolerate=5e-6): #small changes
    lr = initial_lrate
    if epoch >= 10 and epoch < 50:
        lr = 5e-5 
    elif epoch >= 50 and epoch < 200:
        lr = 1e-5
    elif epoch >= 200: 
        lr = min_lr_to_tolerate
    
    print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
######################### Custom data loaders for PSM, PSM wCTL, and GM wCTL #########################

########################################## Vanilla PSM  ##########################################

class PSM_CustomImageDataset(data.Dataset):
    def __init__(self, datapath=""):
        random.seed(912)
        self.img_dir = datapath
        self.imgs = os.listdir(self.img_dir)
        random.shuffle(self.imgs)  ### Random shuffle pair of images to make sure it's not just next frame

    def __len__(self):
        return int(len(self.imgs)/2)

    def __getitem__(self, idx):

#         if (idx>len(self.imgs)/2):
#             return None,None
#         else:
            image1 = Image.open(os.path.join(self.img_dir,self.imgs[2*idx])).convert('RGB')
            image2 = Image.open(os.path.join(self.img_dir,self.imgs[2*idx+1])).convert('RGB')
            processed1 = preprocess.get_transform(augment=True) 
            img1 = processed1(image1)
            img2 = processed1(image2)
            return img1, img2
        
########################################## End Vanilla PSM  ##########################################        


########################################## PSM with CTL  ########################################## 

def temporal_increase_function(epoch):
    
    if epoch <50:
        return epoch*2
    elif epoch <100:
        return epoch*4
    else:
        return epoch*5

class PSM_wCTL_CustomImageDataset(data.Dataset):   # Function of the epoch
    def __init__(self, datapath="",epoch=""):
        random.seed(912)
        temporal_increase = temporal_increase_function(epoch)
        self.img_dir = datapath
        frames = os.listdir(self.img_dir)
        frames_indices = [int(l.split("frame")[1].split(".")[0]) for l in frames] # prepare temporal
        frames_indices_sorted = sorted(frames_indices)
        num_frames = len(frames_indices_sorted)
        indices_to_load = [[frames_indices_sorted[i],frames_indices_sorted[i+temporal_increase]]
                           if ((i+temporal_increase) < num_frames) else [] 
                           for i in range(num_frames)]
        indices_to_load = list(itertools.chain(*indices_to_load))
        indices_to_load = [l for l in indices_to_load if l in frames_indices_sorted]
        self.imgs = ["frame" + str(l) +".jpg" for l in indices_to_load] # prepare temporal  
    
    def __len__(self):
        return int(len(self.imgs)/2)

    def __getitem__(self, idx):

#         if (idx>len(self.imgs)/2):
#             return None,None
#         else:
            image1 = Image.open(os.path.join(self.img_dir,self.imgs[2*idx])).convert('RGB')
            image2 = Image.open(os.path.join(self.img_dir,self.imgs[2*idx+1])).convert('RGB')
            processed1 = preprocess.get_transform(augment=True) 
            img1 = processed1(image1)
            img2 = processed1(image2)
            return img1, img2
        
########################################## End PSM with CTL  ##########################################    

########################################## GM with CTL  ########################################## 
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def temporal_increase_function(epoch):
    
    if epoch <50:
        return epoch*2
    elif epoch <100:
        return epoch*4
    else:
        return epoch*5
    
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    tries = 2
    for i in range(tries):
        try:
            img = Image.open(path).convert('RGB')
        except OSError as e:
            if i < tries - 1: # i is zero indexed
                continue
            else:
                print(path)
                return None
    return img

class GM_wCTL_CustomImageDataset(data.Dataset):
    def __init__(self, datapath="", loader=default_loader,epoch=""):
        random.seed(912)
        self.alldatalist = []
        fp = open (datapath,"r")
        for line in fp.readlines():
            line = line.strip()
            #print(line)
            self.alldatalist.append(line)
        fp.close()
        #print(len(self.alldatalist))

        self.loader = loader
        self.epoch = epoch

    def __getitem__(self, index):
        temporal_increase = temporal_increase_function(self.epoch)
        identity_dir = self.alldatalist[index] + '/'
        split_path = identity_dir.split('/')       
        id_img_list = [identity_dir+'/'+img.name for img in os.scandir(identity_dir) if is_image_file(img.name)]

        #random select one image
        img_idx = np.random.randint(0, len(id_img_list)-1)
        img_idx2 = min(img_idx+temporal_increase,len(id_img_list)-1)       

        img1 = self.loader(id_img_list[img_idx])

        while img1 is None:
            img_idx = np.random.randint(0, len(id_img_list)-1)
            img1 = self.loader(id_img_list[img_idx])

        processed1 = preprocess.get_transform(augment=True)  
        img1 = processed1(img1)
                      
        if np.random.rand() > 0.5:
            img2 = transforms.functional.hflip(img1) 
        else:
            img2 = self.loader(id_img_list[img_idx2])
            while img2 is None:
                img_idx2 = np.random.randint(0, len(id_img_list)-1)      
                img2 = self.loader(id_img_list[img_idx2])  
            processed2 = preprocess.get_transform(augment=True)              
            img2 = processed2(img2)
             
        return img1, img2

    def __len__(self):
        return len(self.alldatalist)

########################################## End GM with CTL  ########################################## 


# Adding the neutral face weight interpolation when specified in the arguments
def forwardloss(im_id0, im_id1, batch,epoch):
    #with torch.autograd.set_detect_anomaly(True):
        optimizer.zero_grad()
        b,c,h,w = im_id0.size()
        # actually runs in batch 32
        full_batch = torch.cat([im_id0, im_id1],dim=0)
            
        expcode = codegeneration(full_batch.data)
        flow_full, backflow_full = exptoflow(expcode)
       
        # generate neutral faces
        neu_face = Swap_Generator(full_batch.data, flow_full)

        # swap face
        split_face = torch.split(neu_face, [b,b], dim=0)
        swap_face = torch.cat([split_face[1], split_face[0]], dim=0)

        # recontruct original faces by direct warp neutral
        direct_forward = warp(full_batch, flow_full)
        split_direct_face = torch.split(direct_forward, [b,b], dim=0)
        swap_direct_face = torch.cat([split_direct_face[1], split_direct_face[0]], dim=0)
        direct_backward = warp(swap_direct_face, backflow_full)
        swap_backward = warp(swap_face, backflow_full)

        # recontruct original faces by warp neutral face features
        recon_face = Swap_Generator(swap_face, backflow_full) #detach or not

        # generator need to recontruct original faces without flow
        rec_img = Swap_Generator(full_batch.data, None)

        #photometric loss        
        pixel_loss = F.l1_loss(recon_face, full_batch.data) + \
                     F.l1_loss(rec_img, full_batch.data) + \
                     F.l1_loss(swap_backward, full_batch.data) + \
                     1.5*F.l1_loss(direct_backward, full_batch.data)  
                
        #percetual loss
        perc_full = torch.cat([full_batch, full_batch, full_batch, full_batch],dim=0) #
        perc_full = perc_full.clone()
        rec_full = torch.cat([neu_face, swap_backward, direct_backward, recon_face],dim=0) #
        im_feat = vgg(perc_full, feat_layers)
        rec_feat = vgg(rec_full, feat_layers)

        pec_loss0 = perceptual_loss(rec_feat[0],im_feat[0]) 
        pec_loss1 = perceptual_loss(rec_feat[1],im_feat[1])
        pec_loss2 = perceptual_loss(rec_feat[2],im_feat[2])

        perc_loss = pec_loss0 + pec_loss1 + pec_loss2

        # SSIM loss
        s_loss = ssim_loss(full_batch, recon_face) + \
                 ssim_loss(full_batch, rec_img) + \
                 1.5*ssim_loss(full_batch, direct_backward)
        
        # Change weight of neutral face loss:
        if args["neu_face_loss_weight"] is not None:
            if(len(args["neu_face_loss_weight"])==1):
                weights = args["neu_face_loss_weight"][0]
                sym_loss = weights*symetricloss(neu_face) + 0.5*symetricloss(direct_forward)
            else:
                weights = np.linspace(args["neu_face_loss_weight"][0],args["neu_face_loss_weight"][1],args["epochs"])
                sym_loss = weights[epoch-1]*symetricloss(neu_face) + 0.5*symetricloss(direct_forward)
        else:
            sym_loss = symetricloss(neu_face) + 0.5*symetricloss(direct_forward)
        
        # neutral face symetric loss
#         sym_loss = symetricloss(neu_face) + 0.5*symetricloss(direct_forward)
        
        loss = 0.01*perc_loss + pixel_loss + 1.5*sym_loss + 5.0*s_loss
        loss.backward()

        optimizer.step()
                                                      
        if batch % 100 == 0:
            print('iter %d percetual loss: %.2f pixel_loss: %.2f ssim loss: %.2f symetric: %.2f' %(batch, perc_loss, pixel_loss, s_loss, sym_loss))

#         if batch % 1000 == 0: 
        if ((epoch % 10 == 0) and (batch ==0)):
            save_image(torch.cat((denorm(full_batch[0:8].data)\
                    , denorm(direct_backward[0:8].data) \
                    , denorm(recon_face[0:8].data)
                    , denorm(neu_face[0:8].data)\
                    ),0), os.path.join(save_image_fold, '{}_{}_decode.png'.format(epoch,int(batch))))

         # Add dataframe with losses:
        if ((epoch % 1 == 0) and (batch ==0)):
            with open(os.path.join(savemodel,"losses.txt"), "a+") as file_object:
                # Move read cursor to the start of file.
                file_object.seek(0)
                # If file is not empty then append '\n'
                data = file_object.read(100)
                if len(data) > 0 :
                    file_object.write("\n")
                # Append text at the end of file
                file_object.write('epoch: %d percetual loss: %.2f pixel_loss: %.2f ssim loss: %.2f symetric: %.2f total loss: %.2f' %(epoch, perc_loss, pixel_loss, s_loss, sym_loss, loss))

                
################## Read the arguments provided by the user to chose the training approach and other parameters: ##################
parser = argparse.ArgumentParser(description='yaml File to use')
parser.add_argument('--yaml_file', default="./Training_approaches/training_arguments.yaml",
                    help='yaml file with specified options for training')
tmp = parser.parse_args()
with open(tmp.yaml_file,'r') as f:
    args = yaml.safe_load(f)
    
args["loadmodel"] = None if args["loadmodel"]=="None" else args["loadmodel"]
args["neu_face_loss_weight"] = None if args["neu_face_loss_weight"]=="None" else args["neu_face_loss_weight"]

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 
torch.manual_seed(2)
torch.cuda.manual_seed(4)


if "GM" in args["custom_data_loader"]: # it means that we are doing GM so no sn
    args["persons_to_train"] = ['']
    
for sn in args["persons_to_train"]:
    print(sn)
    from dataloader import Voxall as DA
    from models import *
    from Loss import *
    import torch.utils.data as data
    from dataloader import *
    torch.cuda.empty_cache()
    if "GM" in args["custom_data_loader"]: # it means that it is GM so no sn
        savemodel = os.path.join(args["savemodel"],args["custom_data_loader"])
        datapath = args["datapath"]
    else:
        savemodel = os.path.join(args["savemodel"],args["custom_data_loader"],sn)
        datapath = os.path.join(args["datapath"],sn,'face_only_aligned')

    if not os.path.exists(savemodel):
        os.makedirs(savemodel)
    else:
        print(savemodel + " exists")
        os.makedirs(os.path.join(savemodel,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        savemodel = os.path.join(savemodel,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(savemodel)
    save_image_fold = os.path.join(savemodel,'imgs/')
    if not os.path.isdir(save_image_fold):
        os.makedirs(save_image_fold)

    with open(os.path.join(savemodel,'arguments.json'), 'w') as f:
            json.dump(str(args), f,  indent=4)


    device = torch.device('cuda')
    vgg = Vgg19(requires_grad=False)

    if torch.cuda.is_available():
        vgg = nn.DataParallel(vgg)
        vgg.to(device)
        vgg.eval()
    feat_layers = ['r21','r31','r41']

    codegeneration =  codegeneration().cuda()
    exptoflow =  exptoflow().cuda()
    Swap_Generator = generator().cuda()

    if args["loadmodel"] is not None:
        state_dict = torch.load(args["loadmodel"])
        Swap_Generator.load_state_dict(state_dict['Swap_Generator'])
        codegeneration.load_state_dict(state_dict['codegeneration'])
        exptoflow.load_state_dict(state_dict['exptoflow'])

    #codegeneration =  nn.DataParallel(codegeneration)
    #exptoflow =  nn.DataParallel(exptoflow)
    #Swap_Generator = nn.DataParallel(Swap_Generator)

    optimizer =  optim.Adam([{"params":Swap_Generator.parameters()},
                             {"params":codegeneration.parameters()},{"params":exptoflow.parameters()}],
                             lr=1e-5, betas=(0.5, 0.999))
    pytorch_total_params = sum(p.numel() for p in codegeneration.parameters())

    if __name__ == '__main__':
        for epoch in range(1,args["epochs"]+1):
            adjust_learning_rate(optimizer, epoch,
                                 initial_lrate=args["initial_lrate"],min_lr_to_tolerate=args["min_lr_to_tolerate"])
            
            ##################################  Prepare the dataloaders for the training approaches.  
            ########################### It has to be inside the training because for CTL it depends on the current epoch
            
                                                        # PSM
            if args["custom_data_loader"]=="PSM":
                TrainImgLoader = torch.utils.data.DataLoader(PSM_CustomImageDataset(datapath=datapath),
                         batch_size=args["batch_size"], shuffle= True,
                         pin_memory=True, collate_fn=fast_collate, num_workers= args["num_workers"], drop_last=True)

                                                                # PSM_wCTL    
            elif args["custom_data_loader"]=="PSM_wCTL":
                TrainImgLoader = torch.utils.data.DataLoader(PSM_wCTL_CustomImageDataset(datapath=datapath,epoch=epoch),
                                                             batch_size=args["batch_size"], shuffle= True,pin_memory=True,
                                     collate_fn=fast_collate, num_workers= args["num_workers"], drop_last=True)

                                                                # GM_wCTL
            elif args["custom_data_loader"]=="GM_wCTL":
                TrainImgLoader = torch.utils.data.DataLoader(GM_wCTL_CustomImageDataset(datapath=datapath,epoch=epoch),
                                                             batch_size=args["batch_size"], shuffle= True,pin_memory=True,
                             collate_fn=fast_collate, num_workers= args["num_workers"], drop_last=True)        

                                                                # GM (Original dataloader from Chang et al.)
            elif args["custom_data_loader"]=="GM":           
                TrainImgLoader = torch.utils.data.DataLoader(
                     DA.myImageloader(datapath=datapath),
                     batch_size= args["batch_size"],shuffle= True,
                     pin_memory=True, collate_fn=fast_collate, num_workers= args["num_workers"], drop_last=True)

            else:
                raise ValueError('''The training approach is not specified.
                                 Please chose between PSM, PSM_wCTL, GM, GM_wCTL.''')
            
            prefetcher = data_prefetcher(TrainImgLoader)

            input0, input1 = prefetcher.next()

            batch_idx = 0
            while input0 is not None:  
                if torch.cuda.is_available(): # change for memory issues
                    input0, input1 = input0.cuda(), input1.cuda()

                    codegeneration.train()
                    exptoflow.train()
                    Swap_Generator.train()
                    forwardloss(input0, input1, batch_idx,epoch) 
                    batch_idx += 1 
                    input0, input1 = prefetcher.next()

            #SAVE
            if not os.path.isdir(savemodel):
                os.makedirs(savemodel)
            # model.module.state_dict() for nn.dataparallel
            if epoch % 10 ==0:
                savefilename = os.path.join(savemodel,'ExpCode_'+str(epoch)+'.tar')
                torch.save({'codegeneration':codegeneration.state_dict(),
                            'exptoflow':exptoflow.state_dict(),
                            'Swap_Generator':Swap_Generator.state_dict(),              
                }, savefilename)
