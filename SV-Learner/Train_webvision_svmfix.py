from __future__ import print_function
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_webvision_svmfix as dataloader
from util_svmfix_webvision import svmfix_train_webvision
from sklearn.mixture import GaussianMixture
from InceptionResNetV2 import InceptionResNetV2
import copy 
import torchnet
from PreResNet_clothing1M import *
from Contrastive_loss import *



parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--id', default='webvision')
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--data_path', default='/data/liangxin/webvision/', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--dataset', default="webvision", type=str)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--sample_threshold', default=0.5, type=float)
parser.add_argument('--lambda_tri', default=0.005, type=float)
parser.add_argument('--flex_threshold', default=0.5, type=float)
parser.add_argument('--gamma', default=2, type=float)
parser.add_argument('--epoch_start_svmfix', default=3, type=int)

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


## For plotting the logs
# import wandb
# wandb.init(project="noisy-label-project-clothing1M", entity="...")

def warmup(net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _ , outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        # penalty = conf_penalty(outputs)
        L = loss # + penalty
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t CE-loss: %.4f'
                         % (args.id, epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))
        sys.stdout.flush()

def test(net1,net2,test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1  = net1(inputs)       
            _, outputs2  = net2(inputs)           
            outputs      = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            

            acc_meter.add(outputs,targets)
    accs = acc_meter.value()

    return accs

## Calculate the KL Divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen_Shannon divergence (Symmetric and Smoother than the KL divergence) 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

## Calculate JSD
def Calculate_JSD(epoch, model1, model2):
    model1.eval()
    model2.eval()
    prob = torch.zeros(len(eval_loader.dataset))
    JS_dist = Jensen_Shannon()
    paths = []
    n=0
    for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda() 
        batch_size      = inputs.size()[0]

        ## Get the output of the Model
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])

        ## Get the Prediction
        out = 0.5*out1 + 0.5*out2          

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
        prob[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist

        sys.stdout.write('\r')
        sys.stdout.write('| Evaluating loss Iter %3d\t' % (batch_idx))
        sys.stdout.flush()

    return prob


def eval_train(model, all_loss):
    model.eval()
    num_iter = (len(eval_loader.dataset) // eval_loader.batch_size) + 1
    losses = torch.zeros(len(eval_loader.dataset))

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' % (batch_idx, num_iter))
            sys.stdout.flush()

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)
    losses = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss

## Penalty for Asymmetric Noise    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

## Get the pre-trained model                
# def get_model():
#     model = InceptionResNetV2(num_classes=args.num_class)
#     model.fc = nn.Linear(2048, args.num_class)
#     return model
#
# def create_model():
#     model = InceptionResNetV2(num_classes=args.num_class)
#     model = model.cuda()
#     return model

def get_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.num_class)
    return model

def create_model():
    model = resnet50(num_classes=args.num_class)
    model = model.cuda()
    return model


loader = dataloader.webvision_dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=8,
                                          imagenet_root='/data/liangxin/openselfsup/data/')
print('| Building Net')

model = get_model()
net1  = create_model()
net2  = create_model()
cudnn.benchmark = True

## Optimizer and Learning Rate Scheduler 
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)


## Cross-Entropy and Other Losses
CE     = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
criterion_triplet = nn.MarginRankingLoss(margin=10.)
contrastive_criterion = SupConLoss()

warm_up = 1

## Location for saving the models 
folder = 'webvision'+ '0.9-1 + w.o SimCLR + batch32'
model_save_loc = './checkpoint/' + '/webvision/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

log = open(model_save_loc +'/%s_'%(args.dataset)+'_acc.txt','w')
log.flush()


net1 = nn.DataParallel(net1)
net2 = nn.DataParallel(net2)

## Loading Saved Weights
model_name_1 = 'webvision_net1.pth.tar'
model_name_2 = 'webvision_net2.pth.tar'

if args.resume:
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1)))
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2)))


best_acc = 0
SR = 0
all_loss = [[],[]] # save the history of losses from two networks
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)


classwise_acc_net1 = torch.zeros((args.num_class,)).cuda()
classwise_acc_net2 = torch.zeros((args.num_class,)).cuda()

class_flex_adjust1 = torch.ones((args.num_class,)).cuda()
class_flex_adjust2 = torch.ones((args.num_class,)).cuda()

gmm_cof1 = 1
gmm_cof2 = 1

for epoch in range(0, args.num_epochs+1):

    lr = args.lr
    if epoch >= 40:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    eval_loader = loader.run(0.5, 'eval_train')
    web_valloader = loader.run(0, 'test')
    imagenet_valloader = loader.run(0, 'imagenet')

    if epoch<warm_up:
        train_loader = loader.run(0,'warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)
        
        print('\nWarmup Net2')
        train_loader = loader.run(0,'warmup')
        warmup(net2, optimizer2, train_loader)

    else:
        print("class_flex_adjust1", class_flex_adjust1)
        print("classwise_acc_net1", classwise_acc_net1)

        num_samples = len(eval_loader.dataset)
        print("num_samples", num_samples)
        prob_js1 = Calculate_JSD(epoch, net1, net2)  # Calculate the JSD distances
        threshold  = torch.mean(prob_js1)                                           ## Simply Take the average as the threshold
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(prob_js1))/args.tau

        SR_js = torch.sum(prob_js1 < threshold).item()/ num_samples # prob_js1.size()[0]

        prob_gmm1, all_loss[1] = eval_train(net2, all_loss[1])
        SR_gmm = np.sum(prob_gmm1 > args.sample_threshold) / num_samples # * gmm_cof1
        # SR = SR_js 
        SR = (SR_js + SR_gmm) / 2.0
        print("gmm_cof1:", gmm_cof1)
        print("SR_gmm:", SR_gmm)
        print("SR_js:", SR_js)
        print("SR:", SR)

        print('\n\nTrain Net1')
        print("class_flex_adjust1", class_flex_adjust1)
        print("threshold", threshold)
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob_js1, class_flex_adjust=class_flex_adjust1, gmm_prob=prob_gmm1)         ## Selection
        print("prob_js12", prob_js1)

        classwise_acc_net1, class_flex_adjust1, gmm_cof1 = svmfix_train_webvision(args, epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, criterion_triplet,
                classwise_acc_net1, contrastive_criterion)   # Train Net1

        print('\n==== Net 1 evaluate next epoch training data loss ====')
        prob_js2 = Calculate_JSD(epoch, net2, net1)  ## Calculate the JSD distances
        threshold = torch.mean(prob_js2)  ## Simply Take the average as the threshold
        if threshold.item() > args.d_u:
            threshold = threshold - (threshold - torch.min(prob_js2)) / args.tau

        SR_js = torch.sum(prob_js2 < threshold).item() / num_samples  # prob_js2.size()[0]

        prob_gmm2, all_loss[0] = eval_train(net1, all_loss[0])
        SR_gmm = np.sum(prob_gmm2 > args.sample_threshold) / num_samples # * gmm_cof2
        # SR = SR_js 
        SR = (SR_js + SR_gmm) / 2.0
        print("gmm_cof1:", gmm_cof2)
        print("SR_gmm:", SR_gmm)
        print("SR_js:", SR_js)
        print("SR:", SR)

        print('\nTrain Net2')
        print("prob_js2", prob_js2)
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob_js2, class_flex_adjust=class_flex_adjust2, gmm_prob=prob_gmm2)

        classwise_acc_net2, class_flex_adjust2, gmm_cof2 = svmfix_train_webvision(args, epoch, net2, net1, optimizer2,
                                                                   labeled_trainloader, unlabeled_trainloader,
                                                                   criterion_triplet,
                                                                   classwise_acc_net2, contrastive_criterion)      ## Train net2
    web_acc = test(net1, net2, web_valloader)
    imagenet_acc = test(net1, net2, imagenet_valloader)

    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n" % (
    epoch, web_acc[0], web_acc[1], imagenet_acc[0], imagenet_acc[1]))
    log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n' % (
    epoch, web_acc[0], web_acc[1], imagenet_acc[0], imagenet_acc[1]))
    if web_acc[0] > best_acc:
        print('| Saving Best Net...')
        torch.save(net1.state_dict(), os.path.join(model_save_loc, model_name_1))
        torch.save(net2.state_dict(), os.path.join(model_save_loc, model_name_2))
        best_acc = web_acc[0]
    log.flush()

