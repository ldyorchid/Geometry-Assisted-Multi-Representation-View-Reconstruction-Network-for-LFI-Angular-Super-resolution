import time
import argparse
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
from os.path import join
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import DatasetFromHdf5
from model import Net
# Training settings
parser =argparse.ArgumentParser(description="PyTorch Light Field Hybrid SR")

#training settings
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=500, help="Learning rate decay every n epochs") # 学习率下降间隔数 每500次将学习率调整为lr*reduce
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=96, help="Training patch size")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="Resume from checkpoint epoch")
parser.add_argument("--num_cp", type=int, default=1, help="Number of epochs for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")
parser.add_argument("--dataset", type=str, default="HCI", help="Dataset for training")
# parser.add_argument("--dataset_path", type=str, default="./LFData/train_SIG.h5")
parser.add_argument("--dataset_path", type=str, default="./LFData/train_HCI.h5")
parser.add_argument("--angular_out", type=int, default=7, help="angular number of the dense light field")
parser.add_argument("--angular_in", type=int, default=2, help="angular number of the sparse light field [AngIn x AngIn]")
opt = parser.parse_args()
print(opt)
#--------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
#--------------------------------------------------------------------------#
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

opt.num_source = opt.angular_in * opt.angular_in
model_dir = 'model_{}_S{}'.format(opt.dataset, opt.num_source)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
#--------------------------------------------------------------------------#
# Data loader
print('===> Loading datasets')
# dataset_path = join('LFData', 'train_{}.h5'.format(opt.dataset))
train_set = DatasetFromHdf5(opt)
train_loader = DataLoader(dataset=train_set,batch_size=opt.batch_size,shuffle=True)
print('loaded {} LFIs from {}'.format(len(train_loader), opt.dataset_path))
#--------------------------------------------------------------------------#
# Build model
print("building net")

model = Net(opt).to(device)
#-------------------------------------------------------------------------#
# optimizer and loss logger
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
losslogger = defaultdict(list)
#------------------------------------------------------------------------#    
# optionally resume from a checkpoint
if opt.resume_epoch:
    resume_path = join(model_dir,'model_epoch_{}.pth'.format(opt.resume_epoch))
    # resume_path = join(model_dir, 'model_epoch.pth')
    if os.path.isfile(resume_path):
        print("==>loading checkpoint 'epoch{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        losslogger = checkpoint['losslogger']
    else:
        print("==> no model found at 'epoch{}'".format(opt.resume_epoch))


#------------------------------------------------------------------------#
# loss
def reconstruction_loss(X,Y):
# L1 Charbonnier loss
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt( diff * diff + eps )
    loss = torch.sum(error) / torch.numel(error)
    return loss
#-----------------------------------------------------------------------#

def train(epoch):

    model.train()
    # scheduler.step()
    loss_count = 0.

    for k in range(10):
         for i, batch in enumerate(train_loader, 1):
            ind_source, input, label, LFI = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            pred_views, pred_lf = model(ind_source, input, LFI, opt)
            loss = reconstruction_loss(pred_lf, label)
            for i in range(pred_views.shape[2]):
                loss += reconstruction_loss(pred_views[:, :, i, :, :], label)
            loss_count += loss.item()
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    scheduler.step()
    losslogger['epoch'].append(epoch)
    losslogger['loss'].append(loss_count/len(train_loader))
    return loss_count/len(train_loader)

# #-------------------------------------------------------------------------#
print('==>training')
min=10
for epoch in range(opt.resume_epoch+1, 3000):
    loss = train(epoch)
    with open("./loss.txt", "a+") as f:
        f.write(str(epoch))
        f.write("\t")
        f.write(str(loss))
        f.write("\t")
        tim = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        f.write(str(tim))
        f.write("\n")

#     checkpoint
    if epoch % opt.num_cp == 0:
        model_save_path = join(model_dir,"model_epoch_{}.pth".format(epoch))
        state = {'epoch':epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'losslogger': losslogger,}
        torch.save(state, model_save_path)
        if min > loss:
            min = loss
            print("update")
            print(min)
            print(epoch)
            print("checkpoint saved to {}".format(model_save_path))

    # loss snapshot
    if epoch % opt.num_snapshot == 0:
        plt.figure()
        plt.title('loss')
        plt.plot(losslogger['epoch'],losslogger['loss'])
        plt.savefig(model_dir+".jpg")
        plt.close()
        

