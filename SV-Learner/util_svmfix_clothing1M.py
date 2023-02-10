import sys
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from random import sample
from collections import Counter
import random


def cal_support_vectors(Lx,net2,inputs_x_w,targets_x,args,batch_size):

    support_vector_h = []
    support_label_h = []
    support_vector_s = []
    support_label_s = []

    # pos_samples = []
    con_x_w2, _ = net2(inputs_x_w)

    images_select_lists_h = [[] for i in range(args.num_class)]
    for i in range(batch_size):
        images_select_lists_h[targets_x[i]].append(i)
    # find support_vector_features
    for j in range(args.num_class):
        if images_select_lists_h[j]:
            _, center_index = torch.topk(Lx[images_select_lists_h[j]], k=1, largest=False)

            index_h = images_select_lists_h[j][center_index]
            support_vector_h.append(con_x_w2[index_h].unsqueeze(0))
            support_label_h.append(j)

            _, sup_index = torch.topk(Lx[images_select_lists_h[j]], k=1, largest=True)
            index_s = images_select_lists_h[j][sup_index]
            support_vector_s.append(con_x_w2[index_s].unsqueeze(0))
            support_label_s.append(j)
        else:
            pass

    support_vector_h = torch.cat(support_vector_h)
    support_vector_h = nn.functional.normalize(support_vector_h, dim=1)

    support_vector_s = torch.cat(support_vector_s)
    support_vector_s = nn.functional.normalize(support_vector_s, dim=1)

    return support_vector_h, support_vector_s, support_label_h, support_label_s



def cal_svmtripet_loss_ori(support_vectors_h, support_vectors_s, pred_norm, target_norm, args, labels, support_labels_h, criterion_triplet, support_labels_s):

    n = pred_norm.size(0)
    dist = -torch.matmul(pred_norm, target_norm.t())

    idx = torch.arange(n)
    mask = idx.expand(n, n).eq(idx.expand(n, n).t())

    # dist_supcon = -torch.matmul(pred_norm, pos_samples.t())
    dist_sup = -torch.matmul(pred_norm, support_vectors_h.t())

    dist_ap, dist_an = [], []
    for i in range(n):

        # dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        pos_label = labels[i]

        neg_class_ids = set(support_labels_h)
        neg_class_ids.remove(int(pos_label))
        neg_class_id = sample(neg_class_ids, 1)

        if support_vectors_h.size(0) < args.num_class:
            # neg_class_id = torch.Tensor(neg_class_id)
            # neg_class_id = int(neg_class_id[0])
            support_labels = np.array(support_labels_h)

            index_neg = np.nonzero(support_labels == neg_class_id[0])
            dist_an.append(dist_sup[i][index_neg])

            if int(pos_label) in support_labels:
                index_pos = np.nonzero(support_labels == int(pos_label))
                # dist_ap.append(dist_supcon[i][index_pos])
                dist_ap.append(dist_sup[i][index_pos])
            else:
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))

        else:

            support_labels = np.array(support_labels_h)
            index_pos = np.nonzero(support_labels == pos_label.item())
            dist_ap.append(dist_sup[i][index_pos])
            # dist_ap.append(dist_supcon[i][index_pos])
            dist_an.append(dist_sup[i][neg_class_id])
        # neg_sup_vec = (torch.sum(dist[i]) - dist[i][pos_label]).mean()

    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    y = torch.ones_like(dist_an)

    loss_triplet = criterion_triplet(dist_an, args.gamma * dist_ap, y)

    return loss_triplet

def cal_svmtripet_loss_hs(support_vectors_h, support_vectors_s, pred_norm, target_norm, args, labels, support_labels_h, criterion_triplet, support_labels_s):

    n = pred_norm.size(0)
    dist = -torch.matmul(pred_norm, target_norm.t())

    idx = torch.arange(n)
    mask = idx.expand(n, n).eq(idx.expand(n, n).t())

    # dist_supcon = -torch.matmul(pred_norm, pos_samples.t())
    dist_sup_h = -torch.matmul(pred_norm, support_vectors_h.t())
    dist_sup_s = -torch.matmul(pred_norm, support_vectors_s.t())
    dist_ap, dist_an = [], []
    for i in range(n):

        # dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        pos_label = labels[i]

        if support_vectors_h.size(0) < args.num_class:
            # neg_class_id = torch.Tensor(neg_class_id)
            # neg_class_id = int(neg_class_id[0])

            support_labels_h = np.array(support_labels_h)
            if int(pos_label) in support_labels_h:
                index_pos = np.nonzero(support_labels_h == int(pos_label))
                # dist_ap.append(dist_supcon[i][index_pos])
                dist_ap.append(dist_sup_h[i][index_pos])
            else:
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))

        else:
            support_labels_h = np.array(support_labels_h)
            index_pos = np.nonzero(support_labels_h == pos_label.item())
            dist_ap.append(dist_sup_h[i][index_pos])
            # dist_ap.append(dist_supcon[i][index_pos])
            # dist_an.append(dist_sup[i][neg_class_id])
        # neg_sup_vec = (torch.sum(dist[i]) - dist[i][pos_label]).mean()

        neg_class_ids = set(support_labels_s)
        if int(pos_label) in neg_class_ids:
            neg_class_ids.remove(int(pos_label))

        if len(neg_class_ids) == 0:
            select_num = random.randint(1, n-1)
            dist_an.append(dist[i][mask[select_num]].max().unsqueeze(0))
        else:
            neg_class_id = sample(neg_class_ids, 1)
            support_labels_s = np.array(support_labels_s)
            index_neg = np.nonzero(support_labels_s == neg_class_id[0])
            dist_an.append(dist_sup_s[i][index_neg])

    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    y = torch.ones_like(dist_an)

    loss_triplet = criterion_triplet(dist_an, args.gamma * dist_ap, y)

    return loss_triplet


def linear_rampup(args, current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


def svmfix_train_clothing1M(args, epoch, net, net2, optimizer, labeled_trainloader,unlabeled_trainloader, criterion_triplet, classwise_acc, contrastive_criterion):

    net2.eval()  # Freeze one network and train the other
    net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    ## Loss statistics
    loss_x = 0
    loss_u = 0
    loss_tri = 0
    loss_ucl = 0

    selected_label = torch.ones((len(unlabeled_trainloader.dataset),), dtype=torch.long, ) * -1
    selected_label = selected_label.cuda()

    selected_label_x = torch.ones((len(labeled_trainloader.dataset),), dtype=torch.long, ) * -1
    selected_label_x = selected_label_x.cuda()

    classwise_acc_next = torch.zeros((args.num_class,)).cuda()

    class_flex_adjust = torch.zeros((args.num_class,)).cuda()

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, index_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4, index = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4, index = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()

        with torch.no_grad():
            # Label co-guessing of unlabeled samples
            _, outputs_u11 = net(inputs_u)
            _, outputs_u12 = net(inputs_u2)
            _, outputs_u21 = net2(inputs_u)
            _, outputs_u22 = net2(inputs_u2)

            ## Pseudo-label
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21,
                                                                                                        dim=1) + torch.softmax(
                outputs_u22, dim=1)) / 4

            ptu = pu ** (1 / args.T)  ## Temparature Sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            max_probs, label_u = torch.max(targets_u, dim=-1)

            if epoch >= 2:
                mask = max_probs.ge(args.flex_threshold).float()
            else:
                mask = max_probs.ge(
                    args.flex_threshold * (classwise_acc[label_u] / (2. - classwise_acc[label_u]))).float()

            mask_idx = (mask == 1).nonzero(as_tuple=False).squeeze(1)

            select = max_probs.ge(args.flex_threshold).long()
            if index[select == 1].nelement() != 0:
                selected_label[index[select == 1]] = label_u.long()[select == 1]

            # Label refinement
            _, outputs_x = net(inputs_x)
            _, outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            px_new = w_x * labels_x + (1 - w_x) * px
            ptx = px_new ** (1 / args.T)  # Temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()
            max_probs_x, label_x = torch.max(targets_x, dim=-1)


            max_probs_true, label_true = torch.max(px, dim=-1)
            select_x = max_probs_true.ge(0.5).long()
            if index_x[select_x == 1].nelement() != 0:
                selected_label_x[index_x[select_x == 1]] = label_x.long()[select_x == 1]

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3[mask_idx], inputs_u4[mask_idx]], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u[mask_idx], targets_u[mask_idx]], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        # Mixup
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        _, logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]

        logits_u = logits[batch_size * 2:]

        probs_u = torch.softmax(logits_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size * 2], dim=1))
        # pure cls
        # _, pure_x_logit = net(inputs_x2)
        # Lx_pure = -torch.mean(torch.sum(F.log_softmax(pure_x_logit, dim=1) * targets_x, dim=1))
        # Lx = 0.95 * Lx_mixup + 0.05 * Lx_pure

        Lu = torch.mean((probs_u - mixed_target[batch_size * 2:]) ** 2)

        lambda_u = linear_rampup(args, epoch + batch_idx / num_iter, args.warm_up)

        ## Regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        # # Unsupervised Contrastive Loss
        # f1, _ = net(inputs_u3)
        # f2, _ = net(inputs_u4)
        # f1 = F.normalize(f1, dim=1)
        # f2 = F.normalize(f2, dim=1)
        # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # loss_simCLR = contrastive_criterion(features)

        if epoch >= args.epoch_start_svmfix:

            with torch.no_grad():
                # sup
                Lx_nogrident = F.cross_entropy(outputs_x, label_x.long(), reduction='none').detach()

                support_vector_h, support_vector_s, support_label_h, support_label_s = cal_support_vectors(Lx_nogrident,
                                                                                net2, inputs_x, label_x, args, batch_size)

                con_u_w2, _ = net2(inputs_u[mask_idx])
                con_u_w2 = nn.functional.normalize(con_u_w2, dim=1)
                # select_con_u_w2 = con_u_w2[select_last_idx_u]
                # select_con_u_w2 = nn.functional.normalize(select_con_u_w2, dim=1)

                con_x_w2, _ = net2(inputs_x)
                con_x_w2 = nn.functional.normalize(con_x_w2, dim=1)

            con_x_s1, _ = net(inputs_x3)
            con_x_s1 = F.normalize(con_x_s1, dim=1)
            con_u_s1, _ = net(inputs_u4[mask_idx])
            con_u_s1 = F.normalize(con_u_s1, dim=1)

            Lc_s = cal_svmtripet_loss_hs(support_vector_h, support_vector_s, con_x_s1, con_x_w2,
                                         args, label_x, support_label_h, criterion_triplet, support_label_s)

            Lc_u = cal_svmtripet_loss_hs(support_vector_h, support_vector_s, con_u_s1, con_u_w2,
                                         args, label_u, support_label_h, criterion_triplet, support_label_s)

            L_hs = (Lc_s + Lc_u) / 2.0

            ## Total Loss
            # loss = Lx + penalty + (args.lambda_tri * L_hs + args.lambda_c * loss_simCLR) / 2.0
            loss = Lx + lambda_u * Lu + penalty + args.lambda_tri * L_hs
            loss_tri += L_hs.item()
        else:
            # Unsupervised Contrastive Loss
            f1, _ = net(inputs_u3)
            f2, _ = net(inputs_u4)
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_simCLR = contrastive_criterion(features)

            loss = Lx + lambda_u * Lu + args.lambda_c * loss_simCLR + penalty
            loss_ucl += loss_simCLR.item()
        # Accumulate Loss
        loss_x += Lx.item()

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            '%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f SimClR Loss:%.4f svmtri Loss:%.4f'
            % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
               loss_x / (batch_idx + 1), loss_ucl / (batch_idx + 1), loss_tri / (batch_idx + 1)))
        sys.stdout.flush()

    pseudo_counter = Counter(selected_label.tolist())
    # wo_negative_one = deepcopy(pseudo_counter)
    # if -1 in wo_negative_one.keys():
    #     wo_negative_one.pop(-1)
    if max(pseudo_counter.values()) < len(unlabeled_trainloader.dataset):  # not all(5w) -1
        for i in range(args.num_class):
            classwise_acc_next[i] = pseudo_counter[i] / max(pseudo_counter.values())
            if classwise_acc_next[i] < 0.6:
                classwise_acc_next[i] = 0.6

    pseudo_counter_x = Counter(selected_label_x.tolist())
    if max(pseudo_counter_x.values()) < len(labeled_trainloader.dataset):  # not all(5w) -1
        avg_confidence = 0
        for i in range(args.num_class):
            class_flex_adjust[i] = pseudo_counter_x[i] / max(pseudo_counter_x.values())
            avg_confidence += class_flex_adjust[i]

            if class_flex_adjust[i] < 0.9:
                class_flex_adjust[i] = 1
        avg_confidence = avg_confidence / args.num_class

    return classwise_acc_next, class_flex_adjust, avg_confidence
