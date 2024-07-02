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

ds = SADIRData(args['data_dir'], test_flag=True)
datal= torch.utils.data.DataLoader(
    ds,
    batch_size=args['batch_size'],
    shuffle=True)
data = iter(datal)
print("number of files: ", len(list(datal)))
temp = torch.cuda.FloatTensor(args['batch_size'], 3, IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE).fill_(0).contiguous()

## self-attention unet
self_model = sadir_diffusion.networks.SADIR_net.load('./results/trained_models/self/self_0500.pt', device)
self_model.to(device)

model = sadir_diffusion.networks.SADIR_net.load('./results/trained_models/gated/gated_0500.pt', device)
model.to(device)        

gts=[]
preds=[]
for _ in range(len(list(datal))):
    try:
        prior, fname = next(data)
    except:
        data = iter(datal)
        prior, fname = next(data)
    time_ = random.randint(100,999) 
    
    # generate image after forward diffusion
    x_t = torch.randn_like(prior[0][0].unsqueeze(0).unsqueeze(0)).cuda()

    for t_ in range(time_, 1, -1):
        inputs = torch.cat([prior.cuda(), x_t], dim=1)
        inputs = [d.cuda().permute(0, 1, 2, 3, 4) for d in inputs.unsqueeze(0)]
        # run inputs through the model to produce a momentum field
        m0_pred = self_model([*inputs, torch.tensor([t_]).cuda()])       
        x0_pred_prc= get_deformed_image(m0_pred, prior[0][1].unsqueeze(0).unsqueeze(0).cuda()).squeeze().unsqueeze(0).unsqueeze(0) 
        x_t = torch.clone(x0_pred_prc)

    ## gated attention unet
    inputs = torch.cat([prior.cuda(), x_t], dim=1)
    inputs = [d.cuda().permute(0, 1, 2, 3, 4) for d in inputs.unsqueeze(0)]
    # run inputs through the model to produce a momentum field
    m0_pred = model([*inputs, torch.tensor([t_]).cuda()])       
    x0_pred_prc= get_deformed_image(m0_pred, prior[0][1].unsqueeze(0).unsqueeze(0).cuda()).squeeze().unsqueeze(0).unsqueeze(0) 
    
    yim= nib.load(args['data_dir']+'test_64/'+fname[0])    
    x0_pred_prc=x0_pred_prc.detach().cpu().numpy().squeeze()
    k = (np.amax(x0_pred_prc) + np.amin(x0_pred_prc))/2
    x0_pred_prc[x0_pred_prc>=k]=1
    x0_pred_prc[x0_pred_prc<k]=0
    nib.save(nib.Nifti1Image(x0_pred_prc, yim.affine,yim.header), './results/predictions/volumes/'+str(fname[0]))
    
    gts.append(yim.get_fdata())
    preds.append(x0_pred_prc)
    
    
# plot results
f, ax = plt.subplots(2,3, figsize=(10,6))
for i in range(2):
    ax[i][0].imshow(prior[0][1].squeeze()[30,:,:], cmap='gray')
    ax[i][0].set_title('Atlas')
    ax[i][0].set_axis_off()
    ax[i][1].imshow(gts[i][30,:,:], cmap='gray')
    ax[i][1].set_title('Ground Truth')
    ax[i][1].set_axis_off()
    ax[i][2].imshow(preds[i][30,:,:], cmap='gray')
    ax[i][2].set_title('Prediction')
    ax[i][2].set_axis_off()
    
plt.savefig('./results/predictions/slices/myocardium.png', dpi=200)
