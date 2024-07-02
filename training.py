import torch, os, glob, gc
os.environ['NEURITE_BACKEND'] = 'pytorch'
from torch.autograd import Variable
from scipy import ndimage
import enum
import torchvision
import math, random
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import sadir_diffusion
import numpy as np
import time
import lagomorph as lm  
from lagomorph import adjrep, deform
import SimpleITK as sitk
from sadir_diffusion.SADIR_forward import get_diffused_image, get_deformed_image, SADIRData  

IMAGE_SIZE=64
device = torch.device('cuda')

args={'data_dir' : './dataset/',
      'batch_size' : 1}
epochs=5
timesteps= random.randint(100,999)

ds = SADIRData(args['data_dir'], test_flag=False)
datal= torch.utils.data.DataLoader(
    ds,
    batch_size=args['batch_size'],
    shuffle=True)
data = iter(datal)
print("number of files: ", len(list(datal)))
temp = torch.cuda.FloatTensor(args['batch_size'], 3, IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE).fill_(0).contiguous()

## self-attention unet

model_dir = './results/trained_models/self/'
self_model = sadir_diffusion.networks.SADIR_net(
        inshape=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE),
        use_attention_unet = False)

# prepare the model for training and send to device
self_model.cuda()
self_model.train()

# set optimizer
optimizer = torch.optim.Adam(self_model.parameters(), lr=1e-3)
grad_loss_func = sadir_diffusion.losses.Grad('l2', loss_mult=1)
image_loss_func = sadir_diffusion.losses.Dice()
v0_loss_func = sadir_diffusion.losses.MSE()

losses = [v0_loss_func, image_loss_func, grad_loss_func]
weights = [1, 0.03, 0.01]

# training loops
for epoch in range(1,epochs):
    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []
    data = iter(datal)
    for step in range(len(list(datal))):
        time_ = random.randint(100,999)
        timesteps=time_  
        prior, x0_true = next(data)
        x_t = get_diffused_image(x0_true, torch.from_numpy(np.array([time_]))).cuda()
        
        for t_ in range(timesteps, 1, -1): 
            inputs = torch.cat([prior.cuda(), x_t], dim=1)
            inputs = [d.cuda().permute(0, 1, 2, 3, 4) for d in inputs.unsqueeze(0)]
            x_0 = [d.cuda().permute(0, 1, 2, 3, 4) for d in x0_true.unsqueeze(0)]
            # run inputs through the model to produce a momentum field
            m0_pred = self_model([*inputs, torch.tensor([t_]).cuda()])       
            x0_pred_prc= get_deformed_image(m0_pred, prior[0][1].unsqueeze(0).unsqueeze(0).cuda()).squeeze().unsqueeze(0).unsqueeze(0) 
            x_t = torch.clone(x0_pred_prc)
                        
        m0_pred.requires_grad_(True)
        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function.loss(x0_pred_prc, x_0[0]) * weights[n]       
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        
    self_model.save(os.path.join(model_dir, 'self_'+'%04d.pt' % epochs))

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, loss_info)), flush=True)

# final model save
self_model.save(os.path.join(model_dir, 'self_'+'%04d.pt' % epochs))


## gated attention unet

device = torch.device("cuda:0")
self_model = sadir_diffusion.networks.SADIR_net.load(os.path.join(model_dir, 'self_'+'%04d.pt' % epochs), device)
self_model.to(device)

model_dir = './results/trained_models/gated/'
model = sadir_diffusion.networks.SADIR_net(
        inshape=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE),
        use_attention_unet = True)

# prepare the model for training and send to device
model.cuda()
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
grad_loss_func = sadir_diffusion.losses.Grad('l2', loss_mult=1)
image_loss_func = sadir_diffusion.losses.Dice()
v0_loss_func = sadir_diffusion.losses.MSE()

losses = [v0_loss_func, image_loss_func, grad_loss_func]
weights = [1, 0.03, 0.01]

# training loops
for epoch in range(epochs):
    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []
    data = iter(datal)
    for step in range(len(list(datal))):
        prior, x0_true = next(data)
        time_ = random.randint(100,999) 
        timesteps= time_  
        x_t = get_diffused_image(x0_true, torch.from_numpy(np.array([time_]))).cuda()
            
        for t_ in range(timesteps, 1, -1): 
            inputs = torch.cat([prior.cuda(), x_t], dim=1)
            inputs = [d.cuda().permute(0, 1, 2, 3, 4) for d in inputs.unsqueeze(0)]
            x_0 = [d.cuda().permute(0, 1, 2, 3, 4) for d in x0_true.unsqueeze(0)]
            # run inputs through the model to produce a momentum field
            with torch.no_grad():
                m0_pred_ = self_model([*inputs, torch.tensor([t_]).cuda()])       
            x0_pred_prc= get_deformed_image(m0_pred_, prior[0][1].unsqueeze(0).unsqueeze(0).cuda()).squeeze().unsqueeze(0).unsqueeze(0) 
            x_t = torch.clone(x0_pred_prc)
       
        inputs = torch.cat([prior.cuda(), x_t], dim=1)
        inputs = [d.cuda().permute(0, 1, 2, 3, 4) for d in inputs.unsqueeze(0)]
        x_0 = [d.cuda().permute(0, 1, 2, 3, 4) for d in x0_true.unsqueeze(0)]
        # run inputs through the model to produce a momentum field
        m0_pred = model([*inputs, torch.tensor([t_]).cuda()])       
        x0_pred_prc= get_deformed_image(m0_pred, prior[0][1].unsqueeze(0).unsqueeze(0).cuda()).squeeze().unsqueeze(0).unsqueeze(0) 
                        
        m0_pred.requires_grad_(True)
        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function.loss(x0_pred_prc, x_0[0]) * weights[n] 
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        
    self_model.save(os.path.join(model_dir, 'gated_'+'%04d.pt' % epochs))

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, loss_info)), flush=True)

# final model save
model.save(os.path.join(model_dir, 'gated_'+'%04d.pt' % epochs))