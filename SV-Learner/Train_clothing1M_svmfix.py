from __future__ import print_function
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
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
import dataloader_clothing1M_svmfix as dataloader
from util_svmfix_clothing1M import svmfix_train_clothing1M
from sklearn.mixture import GaussianMixture
import copy 
import torchnet
from Contrastive_loss import *
from PreResNet_clothing1M import *


parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.005, type=float, help='initial learning rate')   ## Set the learning rate to 0.005 for faster training at the beginning
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=1, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--data_path', default='/data/liangxin/clothing1M/', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=500, type=int)
parser.add_argument('--dataset', default="Clothing1M", type=str)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--sample_threshold', default=0.8, type=float)
parser.add_argument('--lambda_tri', default=0.005, type=float)
parser.add_argument('--flex_threshold', default=0.5, type=float)
parser.add_argument('--warm_up', default=1, type=int)
parser.add_argument('--gamma', default=2, type=float)
parser.add_argument('--epoch_start_svmfix', default=10, type=int)
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
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _ , outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        sys.stdout.flush()
    
def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _ , outputs   = net(inputs)
            _ , predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" %(k,acc))  
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        save_point = os.path.join(model_save_loc, '%s_net%d.pth.tar'%(args.id,k))
        torch.save(net.state_dict(), save_point)
    return acc

def test(net1,net2,test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1  = net1(inputs)       
            _, outputs2  = net2(inputs)           
            outputs      = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total   += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()        
            acc_meter.add(outputs,targets)
            
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    accs = acc_meter.value()
    return acc , accs

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
def Calculate_JSD(epoch,model1, model2):
    model1.eval()
    model2.eval()
    num_samples = args.num_batches*args.batch_size
    prob = torch.zeros(num_samples)
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
        
        for b in range(inputs.size(0)):
            paths.append(path[b])
            n+=1

        sys.stdout.write('\r')
        sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
        sys.stdout.flush()
            
    return prob,paths


def eval_train(model):
    model.eval()
    num_samples = args.num_batches * args.batch_size
    losses = torch.zeros(num_samples)
    paths = []
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[n] = loss[b]
                paths.append(path[b])
                n += 1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' % (batch_idx))
            sys.stdout.flush()

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    losses = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses)
    prob = prob[:, gmm.means_.argmin()]
    return prob

## Penalty for Asymmetric Noise    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

## Get the pre-trained model                
def get_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.num_class)
    return model 

def create_model():
    model = resnet50(num_classes=args.num_class)
    model = model.cuda()
    return model

## Threshold Adjustment 
def linear_rampup(current, warm_up, rampup_length=5):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size, warmup_batch_size = args.batch_size*2, num_workers=8, num_batches=args.num_batches)
print('| Building Net')

model = get_model()
net1  = create_model()
net2  = create_model()
cudnn.benchmark = True

## Optimizer and Learning Rate Scheduler 
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 100, 1e-5)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 100, 1e-5)

## Cross-Entropy and Other Losses
CE     = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
criterion_triplet = nn.MarginRankingLoss(margin=10.)
contrastive_criterion = SupConLoss()

## Warm-up Epochs (maximum value is 2, we recommend 0 or 1)
warm_up = 0

## Copy Saved Data
if args.pretrained: 
    params  = model.named_parameters()
    params1 = net1.named_parameters() 
    params2 = net2.named_parameters()

    dict_params2 = dict(params2)
    dict_params1 = dict(params1)

    for name1, param in params:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param.data)
            dict_params1[name1].data.copy_(param.data)




## Location for saving the models 
folder = 'Clothing1M' + '+batch64_start3_svmfix+ u10 + w-o simclr +classwise1—_st0.85'
# folder = 'Clothing1M'+ 'new0.5conf_batch64_start10_svmfix_js-gmm+g_conf+classwise1—_st0.85'
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

log = open(model_save_loc +'/%s_'%(args.dataset)+'_acc.txt','w')
log.flush()


net1 = nn.DataParallel(net1)
net2 = nn.DataParallel(net2)

## Loading Saved Weights
model_name_1 = 'clothing1m_net1.pth.tar'
model_name_2 = 'clothing1m_net2.pth.tar'

if args.resume:
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1)))
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2)))

best_acc = [0,0]
SR = 0
torch.backends.cudnn.benchmark = True
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
nb_repeat = 2


classwise_acc_net1 = torch.zeros((args.num_class,)).cuda()
classwise_acc_net2 = torch.zeros((args.num_class,)).cuda()

class_flex_adjust1 = torch.ones((args.num_class,)).cuda()
class_flex_adjust2 = torch.ones((args.num_class,)).cuda()


gmm_cof1 = 1
gmm_cof2 = 1
jsd_cof1 = 1
jsd_cof2 = 1

for epoch in range(0, args.num_epochs+1):   
    val_loader = loader.run(0, 'val')
    
    if epoch>100:
        nb_repeat =3  ## Change how many times we want to repeat on the same selection

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

        
        num_samples = args.num_batches*args.batch_size
        eval_loader = loader.run(0.5,'eval_train')
        prob_js1, paths1 = Calculate_JSD(epoch, net1, net2)                          ## Calculate the JSD distances
        threshold   = torch.mean(prob_js1)                                           ## Simply Take the average as the threshold
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(prob_js1))/args.tau
            
        if epoch > args.epoch_start_svmfix:
            jsd_cof1 = (1 - threshold) / (1 - torch.min(prob_js1))
        SR_js = torch.sum(prob_js1<threshold).item()/prob_js1.size()[0]  # * jsd_cof1                  ## Calculate the Ratio of clean samples

        prob_gmm1 = eval_train(net2)
        SR_gmm = np.sum(prob_gmm1 > args.sample_threshold) / num_samples # * gmm_cof1
        # SR = SR_js 
        SR = (SR_js + SR_gmm) / 2.0
        print("gmm_cof1:", gmm_cof1)
        # print("jsd_cof1:", jsd_cof1)
        print("SR_gmm:", SR_gmm)
        print("SR_js:", SR_js)
        print("SR:", SR)

        for i in range(nb_repeat):
            print('\n\nTrain Net1')
            print("class_flex_adjust1", class_flex_adjust1)
            print("threshold", threshold)
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob_js1,  paths=paths1, class_flex_adjust=class_flex_adjust1, gmm_prob=prob_gmm1)         ## Selection
            print("prob_js12", prob_js1)
            if i == nb_repeat-1:
              classwise_acc_net1, class_flex_adjust1, gmm_cof1 = svmfix_train_clothing1M(args, epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader,                                                                    criterion_triplet, classwise_acc_net1, contrastive_criterion)     ## Train Net1
            else:
              classwise_acc_net1, _, gmm_cof1 = svmfix_train_clothing1M(args, epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, criterion_triplet,
                    classwise_acc_net1, contrastive_criterion)     ## Train Net1
            acc1 = val(net1,val_loader,1)

        print('\n==== Net 1 evaluate next epoch training data loss ====') 
        eval_loader   = loader.run(SR,'eval_train')
        net1.load_state_dict(torch.load(os.path.join(model_save_loc, '%s_net1.pth.tar'%args.id)))

        prob_js2, paths2 = Calculate_JSD(epoch, net2, net1)  ## Calculate the JSD distances
        threshold = torch.mean(prob_js2)  ## Simply Take the average as the threshold
        if threshold.item() > args.d_u:
            threshold = threshold - (threshold - torch.min(prob_js2)) / args.tau
            
        if epoch > args.epoch_start_svmfix:
            jsd_cof2 = (1 - threshold) / (1 - torch.min(prob_js2)) 
        SR_js = torch.sum(prob_js2 < threshold).item() / prob_js2.size()[0] # * jsd_cof2 ## Calculate the Ratio of clean samples

        prob_gmm2 = eval_train(net1)
        SR_gmm = np.sum(prob_gmm2 > args.sample_threshold) / num_samples # * gmm_cof2
        # SR = SR_js 
        SR = (SR_js + SR_gmm) / 2.0
        print("gmm_cof1:", gmm_cof2)
        # print("jsd_cof1:", jsd_cof2)
        print("SR_gmm:", SR_gmm)
        print("SR_js:", SR_js)
        print("SR:", SR)

        for i in range(nb_repeat):
            print('\nTrain Net2')
            print("prob_js21", prob_js2)
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob_js2, paths=paths2, class_flex_adjust=class_flex_adjust2, gmm_prob=prob_gmm2)           ## Uniform Selection
            print("prob_js22", prob_js2)
            if i == nb_repeat-1:
              classwise_acc_net2, class_flex_adjust2, gmm_cof2 = svmfix_train_clothing1M(args, epoch, net2, net1, optimizer2,
                                                                       labeled_trainloader, unlabeled_trainloader,
                                                                       criterion_triplet,
                                                                       classwise_acc_net2, contrastive_criterion)                             ## Train net2
            else:
              classwise_acc_net2, _, gmm_cof2 = svmfix_train_clothing1M(args, epoch, net2, net1, optimizer2,
                                                                        labeled_trainloader, unlabeled_trainloader,
                                                                        criterion_triplet,
                                                                        classwise_acc_net2, contrastive_criterion)                             ## Train net2
            acc2 = val(net2,val_loader,2)

    scheduler1.step()
    scheduler2.step()        
    acc1 = val(net1,val_loader,1)
    acc2 = val(net2,val_loader,2)   
    log.write('Validation Epoch:%d  Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))

    net1.load_state_dict(torch.load(os.path.join(model_save_loc, '%s_net1.pth.tar'%args.id)))
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, '%s_net2.pth.tar'%args.id)))
    log.flush() 
    test_loader = loader.run(0,'test')  
    acc, accs = test(net1,net2,test_loader)   
    print('\n| Epoch:%d \t  Acc: %.2f%% (%.2f%%) \n'%(epoch,accs[0],accs[1]))
    log.write('Epoch:%d \t  Acc: %.2f%% (%.2f%%) \n'%(epoch,accs[0],accs[1]))
    log.flush()  

    if epoch<warm_up: 
        model_name_1 = 'Net1_warmup_pretrained.pth'     
        model_name_2 = 'Net2_warmup_pretrained.pth' 

        print("Save the Warmup Model --- --")
        checkpoint1 = {
            'net': net1.state_dict(),
            'Model_number': 1,
            'epoch': epoch,
        }
        checkpoint2 = {
            'net': net2.state_dict(),
            'Model_number': 2,
            'epoch': epoch,
        }
        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))   
