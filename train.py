import os
import time
import torch
import json
import numpy as np
import sklearn.metrics as metrics
from model import GMN
from args import get_parser
from dataset import Dataset
from datetime import datetime
from torch.autograd import Variable
from tensorboard_logger import configure, log_value


def adjust_learning_rate(optimizer1, optimizer2, optimizer3, lr, decay=0.5):
    lr *= decay
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer3.param_groups:
        param_group['lr'] = lr
    return optimizer1, optimizer2, optimizer3, lr


def train(ds, args, mask_nodes=True):
    val_accs = []
    configure(f'{args.logdir}/tensorboard')

    for k in range(10):
        print('*' * 40)
        print(f'Setting up the model for fold {k + 1} ...')

        lr = args.lr
        patience_counter = 0
        best_train_acc, best_val_acc = 0.0, 0.0
        total_num_cluster = len(args.num_centroids)

        model = GMN(0.2, 1, args, ds.max_nodes)
        model = model.cuda()

        print('Model configuration:')
        print(args)

        param_dict = [{'params': model.centroids, 'lr': lr},
                      {'params': list(model.parameters())[1:], 'lr': lr}]
        param_dict_3 = [{'params': list(model.parameters())[1:], 'lr': lr}]

        optimizer1 = torch.optim.Adam(param_dict, lr=lr, weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam([model.centroids], lr=lr, weight_decay=args.weight_decay)
        optimizer3 = torch.optim.Adam(param_dict_3, lr=lr, weight_decay=args.weight_decay)

        ds.process(batch_size=args.batch_size, val_idx=k, normalize_adj=args.normalize_adj)

        print('#Epochs: ', args.num_epochs)

        for epoch in range(args.num_epochs):
            labels_list, preds_list, loss_list, \
            labels_list_val, preds_list_val = \
            [], [], [], [], []

            model.train()
            if ((epoch + 1) % args.decay_step) == 0:
                optimizer1, optimizer2, optimizer3, lr = \
                    adjust_learning_rate(optimizer1, optimizer2, optimizer3, lr, decay=0.5)

            start = time.time()
            for batch_idx, batch in enumerate(ds.train):
                batch_num_nodes = batch['num_nodes'].int().numpy() if mask_nodes else None
                h0 = Variable(batch['feats'].float(), requires_grad=False).cuda()
                label = Variable(batch['label'].long()).cuda()

                if args.use_rwr:
                    adj = Variable(batch['rwr'].float(), requires_grad=False).cuda()
                else:
                    adj = Variable(batch['adj'].float(), requires_grad=False).cuda()

                for c_layer in range(total_num_cluster):
                    if total_num_cluster == 1 or c_layer == 0:
                        new_adj = adj.clone().detach().requires_grad_(False)
                        new_feat = h0.clone().detach().requires_grad_(False)
                        del adj, h0

                    if c_layer != 0:
                        new_adj.requires_grad_(True)
                        new_feat.requires_grad_(True)
                    if c_layer + 1 < total_num_cluster:
                        master_node_flag = False
                        for c_iter in range(args.num_clusteriter):
                            __, output, new_adj, new_feat, __ = \
                            model(new_feat, new_adj, epoch, batch_num_nodes, c_layer, master_node_flag)
                            hard_loss = output

                    else:
                        master_node_flag = True
                        __, __, __, __, h_prime = \
                        model(new_feat, new_adj, epoch, batch_num_nodes, c_layer, master_node_flag)

                preds = torch.squeeze(h_prime)
                loss = model.loss(preds, label)
                model.centroids.requires_grad_(False)
                if (epoch + 1) % args.backward_period == 1 and \
                        len(args.num_centroids) > 1 and \
                        args.learn_centroid is not 'f':
                    model.centroids.requires_grad_(True)
                    if args.learn_centroid == 'c':
                        hard_loss.backward()

                    elif args.learn_centroid == 'a':
                        hard_loss.backward()

                if (epoch+1) % args.backward_period == 1 and \
                        len(args.num_centroids) > 1 and \
                        args.learn_centroid is not 'f':
                    model.centroids.requires_grad_(True)
                else:
                    optimizer3.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer3.step()

                labels_list.append(label.detach().cpu().numpy())
                __, idx = torch.max(preds, 1)

                preds_list.append(idx.detach().data.cpu().numpy())
                loss_list.append(loss.detach().data.cpu().numpy())

            if (epoch+1) % args.backward_period == 1 and \
                    len(args.num_centroids) > 1 and \
                    args.learn_centroid != 'f':
                model.centroids.requires_grad_(True)

                if args.avg_grad:
                    for i, m in enumerate(model.parameters()):
                        if m.grad is not None:
                            list(model.parameters())[i].grad = m.grad / batch_idx
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                if args.learn_centroid == 'c':
                    optimizer2.step()
                    optimizer2.zero_grad()
                elif args.learn_centroid == 'a':
                    optimizer1.step()
                    optimizer1.zero_grad()

            end = time.time()

            with torch.no_grad():
                model.eval()
                for val_batch_idx, batch_val in enumerate(ds.val):
                    batch_num_nodes = batch_val['num_nodes'].int().numpy() if mask_nodes else None
                    h0 = Variable(batch_val['feats'].float(), requires_grad=False).cuda()
                    label = Variable(batch_val['label'].long()).cuda()

                    if args.use_rwr:
                        adj = Variable(batch_val['rwr'].float(), requires_grad=False).cuda()
                    else:
                        adj = Variable(batch_val['adj'].float(), requires_grad=False).cuda()

                    for c_layer in range(total_num_cluster):
                        if total_num_cluster == 1 or c_layer == 0:
                            new_adj = adj.clone().detach().requires_grad_(False)
                            new_feat = h0.clone().detach().requires_grad_(False)
                            del adj, h0

                        if c_layer + 1 < total_num_cluster:
                            master_node_flag = False
                            for c_iter in range(args.num_clusteriter):
                                centroid_tensor, output, new_adj, new_feat, __ = \
                                    model(new_feat, new_adj, epoch, batch_num_nodes, c_layer, master_node_flag)
                        else:
                            master_node_flag = True
                            __, __, __, __, h_prime = \
                                model(new_feat, new_adj, epoch, batch_num_nodes, c_layer, master_node_flag)
                    preds_val = torch.squeeze(h_prime)
                    labels_list_val.append(label.cpu().numpy())
                    __, idx_val = torch.max(preds_val, 1)

                    preds_list_val.append(idx_val.detach().data.cpu().numpy())

            acc_train = metrics.accuracy_score(np.squeeze(np.hstack(labels_list)), np.hstack(preds_list))
            acc_val = metrics.accuracy_score(np.squeeze(np.hstack(labels_list_val)), np.hstack(preds_list_val))
            best_val_acc = acc_val if acc_val > best_val_acc else best_val_acc
            best_train_acc = acc_train if acc_train > best_train_acc else best_train_acc

            if epoch % 1 == 0:
                print('*' * 40)
                print(f'Fold:{k + 1}, Epoch:{epoch + 1}, Time:{end - start:.2f}s')
                print(f'Train loss:{np.mean(loss_list):.4f}')
                print(f'Train accuracy:{acc_train * 100:.2f}%, Validation accuracy:{acc_val * 100:.2f}%')
                print(f'Best train accuracy:{best_train_acc * 100:.2f}%')
                print(f'Best validation accuracy:{best_val_acc * 100:.2f}%')

            if acc_val < best_val_acc:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter == args.patience:
                print('Early stoppling ...')
                break
            log_value('train loss', np.mean(loss_list), epoch + k * args.num_epochs)
            log_value('train accuracy', acc_train, epoch + k * args.num_epochs)
            log_value('val accuracy', acc_val, epoch + k * args.num_epochs)
            log_value('lr', lr, epoch + k * args.num_epochs)

        torch.save(model.state_dict(), f'{args.logdir}/checkpoints/model_fold{k + 1}.pkl')

        val_accs.append(100 * round(best_val_acc, 4))
        acc = str([f'{a:.2f}%' for a in val_accs])
        print('*' * 40)
        print(f'Validation accuracy upto fold {k + 1}: {acc}')
        print(f'Mean validation accuracy upto fold {k + 1}: {np.mean(val_accs):.2f}%')

    return val_accs


def set_seeds():
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    import warnings
    warnings.filterwarnings("ignore")
    #set_seeds()
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_index
    print('CUDA', args.cuda_index)

    datasets = ('ENZYMES', 'DD', 'REDDIT-MULTI-12K', 'COLLAB', 'PROTEINS_full', 'REDDIT-BINARY')
    benchmark = datasets[0]
    args.dataset = benchmark

    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    args.logdir = f'logs/{benchmark}/{now}'

    if not os.path.exists(f'{args.logdir}/checkpoints'):
        os.makedirs(f'{args.logdir}/checkpoints')

    ds = Dataset(name=benchmark, max_nodes=1000, num_folds=10)
    args.input_dim = ds.feat_dim
    args.output_dim = args.input_dim
    args.num_classes = ds.num_class
    args.num_centroids = [int(x) for x in args.num_centroids.split(',') if x.strip().isdigit()]

    val_accs = train(ds, args)
    args.mean_validation_accuracy = np.mean(val_accs)
    args.std_validation_accuracy = np.std(val_accs)
    args.best_fold = int(np.argmax(val_accs))
    args.best_validation_accuracy = np.max(val_accs)
    args.validation_accuracies = val_accs

    with open(f'{args.logdir}/summary.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    main()

