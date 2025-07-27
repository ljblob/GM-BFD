import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from misc import AverageMeter
from models import GMBFD

from test_metrics import test
from utils_var import *
from utils_var import getBatchedEdgeMat 
from collections import Counter
import os
import argparse
from datetime import datetime
from select_pseudo_label import select_pseudo_label


def diff_loss(shared_embedding, task_embedding):
    shared_embedding -= torch.mean(shared_embedding, 0)
    task_embedding -= torch.mean(task_embedding, 0)

    shared_embedding = torch.nn.functional.normalize(shared_embedding, dim=1, p=2)
    task_embedding = torch.nn.functional.normalize(task_embedding, dim=1, p=2)

    correlation_matrix = task_embedding.t() @ shared_embedding
    loss_diff = torch.mean(torch.square_(correlation_matrix)) * 0.01
    loss_diff = torch.where(loss_diff > 0, loss_diff, 0)
    return loss_diff


def main(args):
    # Trian on both source and target domains, only source domain's label is used for trainging
    # Test on both target and unseen target_2 domain
    source_domain_fg = re.findall(r'\d+', args.source_dataset_name)[0]
    target_domain_fg = re.findall(r'\d+', args.target_dataset_name)[0]
    if args.target_2_dataset_name is not None:
        target_2_domain_fg = re.findall(r'\d+', args.target_2_dataset_name)[0]
    else:
        target_2_domain_fg = None


    dataset_root_path = ['./data/CTU-13_preprocessed/']
    vocab_root_path = './data/CTU-13_preprocessed/Vocab/'

    cuda = True
    cudnn.benchmark = True

    # 0. load data
    '''
    ['StartTime', 'Dur', 'Proto', 'SrcAddr', 'Sport', 'Dir', 'DstAddr',
        'Dport', 'State', 'sTos', 'dTos', 'TotPkts', 'TotBytes', 'SrcBytes',
        'Label']
    '''
    if target_2_domain_fg == None:
        src_data, tag_data, vocab = load_files(dataset_root_path[0], vocab_root_path, source_domain_fg, target_domain_fg, args.where_GPU)
    else:
        src_data, tag_data, tag_2_data, vocab = load_files_multiTarget(dataset_root_path[0], vocab_root_path, source_domain_fg, target_domain_fg, target_2_domain_fg, args.where_GPU)
    print('vocab length:', len(create_vocab_dict(vocab, num_to_encode=3)))

    # 1. preprocess data
    str_col = ['Proto', 'Sport', 'Dir',
            'Dport', 'State', 'sTos', 'dTos']

    src_str_data, src_num_tensor, src_label = preprocess_data(src_data, vocab, str_col)  
    tag_str_data, tag_num_tensor, tag_label = preprocess_data(tag_data, vocab, str_col)  
    tag_2_str_data, tag_2_num_tensor, tag_2_label = preprocess_data(tag_2_data, vocab, str_col)
    tag_src_ip_lst = list(tag_data['SrcAddr'])
    tag_dst_ip_lst = list(tag_data['DstAddr'])
    src_src_ip_lst = list(src_data['SrcAddr'])
    src_dst_ip_lst = list(src_data['DstAddr'])
    src_lst = [src_src_ip_lst, src_dst_ip_lst]
    tag_lst = [tag_src_ip_lst, tag_dst_ip_lst]
    
    tag_2_src_ip_lst = list(tag_2_data['SrcAddr'])
    tag_2_dst_ip_lst = list(tag_2_data['DstAddr'])
    tag_2_lst = [tag_2_src_ip_lst, tag_2_dst_ip_lst]
    
    # training
    meters = ['src_pre', 'src_rec', 'src_f1', 'src_accu', 'tag_pre',  'tag_rec',  'tag_f1', 'tag_accu', 
            'tag_2_pre',  'tag_2_rec',  'tag_2_f1', 'tag_2_accu']
    results = {meter: AverageMeter() for meter in meters}
    
    for itr in range(0, args.iterations):
        best_f1_t = 0.0
        best_f1_t2 = 0.0
        # Dataloader
        dataset_source, dataloader_source = get_DataLoader(src_num_tensor, src_str_data, src_label, src_lst,
                                                        batchsize=args.batch_size, shuffle=True, dropLast=True)
        dataset_target, dataloader_target = get_DataLoader(tag_num_tensor, tag_str_data, tag_label, tag_lst,
                                                        batchsize=args.batch_size, shuffle=True, dropLast=True)
        dataset_target_2, dataloader_target_2 = get_DataLoader(tag_2_num_tensor, tag_2_str_data, tag_2_label, tag_2_lst,
                                                            args.batch_size, shuffle=True, dropLast=True)
        # load model
        cls_num = src_data['Label'].max() + 1
        args.cls_num = cls_num
        my_net = GMBFD(src_str_data[1].shape[0] + src_num_tensor.shape[1], cls_num, args.device, args.batch_size,
                        args.gat_hidden_dim, args.num_head)

        # setup optimizer
        optimizer = optim.Adam(my_net.parameters(), lr=args.lr, weight_decay=args.wdecay)

        loss_class = torch.nn.NLLLoss()
        loss_domain = torch.nn.NLLLoss()
        loss_subdomain = torch.nn.NLLLoss()

        if cuda:
            my_net = my_net.to(args.device)
            loss_class = loss_class.to(args.device)
            loss_domain = loss_domain.to(args.device)
            loss_subdomain = loss_subdomain.to(args.device)

        for p in my_net.parameters():
            p.requires_grad = True

        # my_net.zero_grad()
        best_f1_t_current_epoch = 0.0
        best_f1_current_epoch_sum = 0.0
        
        best_f1_t2_current_epoch = 0.0

        for epoch in range(1, args.n_epoch+1):

            len_dataloader = min(len(dataloader_source), len(dataloader_target))
            data_source_iter = iter(dataloader_source)
            data_target_iter = iter(dataloader_target)

            losses = AverageMeter()
            losses_s_label = AverageMeter()
            losses_s_domain = AverageMeter()
            losses_t_domain = AverageMeter()
            loss_s_local = AverageMeter()
            loss_t_local = AverageMeter()
            t_pl_acc_record = AverageMeter()
            s_pl_acc_record = AverageMeter()
            loss_err_diff = AverageMeter()

            for i in range(len_dataloader):

                my_net.train()
                optimizer.zero_grad()

                p = float(i + epoch * len_dataloader) / args.n_epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # training model using source data
                data_source = next(data_source_iter)
                s_num, s_str, s_label, s_indices, s_src_ip, s_des_ip = data_source
                src_edge_mat = getBatchedEdgeMat(args.batch_size, s_src_ip, s_des_ip).to(args.device)

                s_domain_label = torch.zeros(len(s_label)).long()
                s_num, s_str, s_label, s_domain_label, s_indices = \
                    s_num.to(args.device), s_str.to(args.device), s_label.to(args.device), s_domain_label.to(args.device), s_indices.to(args.device)

                src_class_output, src_g_domain_output, src_l_domain_output, src_feature = my_net(input_data_str=s_str,\
                                                                        input_data_num=s_num, alpha=alpha, edge_index=src_edge_mat)

                err_s_label = loss_class(src_class_output, s_label)  # classification loss
                err_s_domain = loss_domain(src_g_domain_output, s_domain_label)  # global loss

                local_loss_s = 0.0  # local loss
                if epoch > args.start_estimate_pl:
                    
                    s_pl_loss, s_pl_acc, s_pl_positive, s_total_select_positive, s_pseudo_label_dict \
                        = select_pseudo_label(args, [s_str, s_num, s_label, s_indices], my_net, itr, 'source', src_edge_mat)
                    s_pseudo_idx = s_pseudo_label_dict['source_pseudo_idx']  
                    s_pseudo_target = s_pseudo_label_dict['source_pseudo_target']  
                    s_pseudo_idx_np = np.array(s_pseudo_idx)
                    s_batch_idx = np.where(np.isin(s_indices.cpu().numpy(), s_pseudo_idx_np))[0] 

                    for c_i in range(args.cls_num):
                        loss_si = loss_subdomain(src_l_domain_output[c_i][s_batch_idx], s_domain_label[s_batch_idx])
                        local_loss_s += loss_si

                    s_pl_acc_record.update(s_pl_acc)

                # training model using target data
                data_target = next(data_target_iter)
                t_num, t_str, t_label, t_indices, t_src_ip, t_des_ip = data_target
                tag_edge_mat = getBatchedEdgeMat(args.batch_size, t_src_ip, t_des_ip).to(args.device)
                
                t_domain_label = torch.ones(len(t_num)).long()
                t_num, t_str, t_domain_label, t_label, t_indices = \
                    t_num.to(args.device), t_str.to(args.device), t_domain_label.to(args.device), \
                        t_label.to(args.device), t_indices.to(args.device)

                trg_class_output, trg_g_domain_output, trg_l_domain_output, trg_feature = my_net(input_data_str=t_str, \
                                                                    input_data_num=t_num, alpha=alpha, edge_index=tag_edge_mat)

                err_t_domain = loss_domain(trg_g_domain_output, t_domain_label)  # global loss
                local_loss_t = 0.0  # local loss

                if epoch > args.start_estimate_pl:
                    t_pl_loss, t_pl_acc, t_pl_positive, t_total_select_positive, t_pseudo_label_dict \
                        = select_pseudo_label(args, [t_str, t_num, t_label, t_indices], my_net, itr, 'target', tag_edge_mat)
                    t_pseudo_idx = t_pseudo_label_dict['target_pseudo_idx'] 
                    t_pseudo_target = t_pseudo_label_dict['target_pseudo_target']  
                    t_pseudo_idx_np = np.array(t_pseudo_idx)
                    t_batch_idx = np.where(np.isin(t_indices.cpu().numpy(), t_pseudo_idx_np))[0] 

                    for c_i in range(args.cls_num):
                        loss_ti = loss_subdomain(trg_l_domain_output[c_i][t_batch_idx], t_domain_label[t_batch_idx])
                        local_loss_t += loss_ti

                    t_pl_acc_record.update(t_pl_acc)

                err_diff = diff_loss(src_feature, trg_feature)

                err = (err_s_label + err_s_domain +err_t_domain) +  (local_loss_s + local_loss_t + args.gamma_loss*err_diff)
                # err = args.loss_alpha*(err_s_label + err_s_domain +err_t_domain) +  (1-args.loss_alpha)*(local_loss_s + local_loss_t + args.gamma_loss*err_diff)

                losses.update(err.item())
                losses_s_label.update(err_s_label.item())
                losses_s_domain.update(err_s_domain.item())
                losses_t_domain.update(err_t_domain.item())
                loss_s_local.update(local_loss_s)
                loss_t_local.update(local_loss_t)
                loss_err_diff.update(err_diff)

                err.backward()
                optimizer.step()
                # my_net.zero_grad()

            # torch.save(my_net, '{0}/model_epoch_current.pth'.format(args.model_root))
            
            # Testing ...
            accu_s, pre_s, rec_s, f1_s, fpr_s = test(dataloader_source, args.device, src_src_ip_lst, src_dst_ip_lst, batch_size=args.batch_size, my_net=my_net)
            print('Iter=%d, Epoch=%d, Metrics for the %s dataset: Accuracy=%.4f, Recall=%.4f, Precision=%.4f, F1=%.4f, FPR=%.4f'
                % (itr, epoch, args.source_dataset_name, accu_s, pre_s, rec_s, f1_s, fpr_s), flush=True)
            accu_t, pre_t, rec_t, f1_t, fpr_t = test(dataloader_target, args.device, tag_src_ip_lst, tag_dst_ip_lst, batch_size=args.batch_size, my_net=my_net)
            print('Iter=%d, Epoch=%d, Metrics for the %s dataset: Accuracy=%.4f, Recall=%.4f, Precision=%.4f, F1=%.4f, FPR=%.4f'
                % (itr, epoch, args.target_dataset_name, accu_t, rec_t, pre_t, f1_t, fpr_t), flush=True)
            
            accu_t2, pre_t2, rec_t2, f1_t2, fpr_t2 = test(dataloader_target_2, args.device, tag_2_src_ip_lst, tag_2_dst_ip_lst, batch_size=args.batch_size, my_net=my_net)
            print('Iter=%d, Epoch=%d, Metrics for the %s dataset: Accuracy=%.4f, Recall=%.4f, Precision=%.4f, F1=%.4f, FPR=%.4f'
            % (itr, epoch, args.target_2_dataset_name, accu_t2, pre_t2, rec_t2, f1_t2, fpr_t2), flush=True)
            
            if epoch > args.start_estimate_pl:
                print('Iter=%d, Epoch=%d, train_loss=%.4f, loss_s_label=%.4f, loss_s_domain=%.4f, loss_t_domain=%.4f, '
                    'loss_s_local=%.4f, loss_t_local=%.4f, s_pl_acc=%.4f, t_pl_acc=%.4f, src_pseudo_target=%s, '
                    'trg_pseudo_target=%s, loss_err_diff=%.4f'
                    % (itr, epoch, losses.avg, losses_s_label.avg, losses_s_domain.avg, losses_t_domain.avg, loss_s_local.avg,
                        loss_t_local.avg, s_pl_acc_record.avg, t_pl_acc_record.avg, str(Counter(s_pseudo_target)),
                        str(Counter(t_pseudo_target)), loss_err_diff.avg))
            else:
                print('Iter=%d, Epoch=%d, train_loss=%.4f, loss_s_label=%.4f, loss_s_domain=%.4f, loss_t_domain=%.4f, '
                    'loss_s_local=%.4f, loss_t_local=%.4f, s_pl_acc=%.4f, t_pl_acc=%.4f, loss_err_diff=%.4f'
                    % (itr, epoch, losses.avg, losses_s_label.avg, losses_s_domain.avg, losses_t_domain.avg, loss_s_local.avg,
                        loss_t_local.avg, s_pl_acc_record.avg, t_pl_acc_record.avg, loss_err_diff.avg))

            if f1_t2 > best_f1_t2_current_epoch:
                best_f1_t2_current_epoch = f1_t2
                best_pre_t2_current_epoch = pre_t2
                best_rec_t2_current_epoch = rec_t2
                best_accu_t2_current_epoch = accu_t2
                best_fpr_t2_current_epoch = fpr_t2
                
            if (f1_t+f1_s) > best_f1_current_epoch_sum:
                best_f1_current_epoch_sum = f1_t + f1_s
                
                best_f1_t_current_epoch = f1_t
                best_pre_t_current_epoch = pre_t
                best_rec_t_current_epoch = rec_t
                best_accu_t_current_epoch = accu_t
                best_fpr_t_current_epoch = fpr_t

                best_f1_s_current_epoch = f1_s
                best_pre_s_current_epoch = pre_s
                best_rec_s_current_epoch = rec_s
                best_accu_s_current_epoch = accu_s
                best_fpr_s_current_epoch = fpr_s
                torch.save(my_net, '{0}/model_epoch_best.pth'.format(args.model_root))    
            
            
            if epoch != 0 and epoch%5 == 0:
                print('\n============Epoch Summary =============')
                print('Iter=%d, Metrics for the %s dataset: Recall: %4f, accu: %4f, Precision: %4f, F1: %4f, FPR: %4f'
                    % (itr, args.source_dataset_name, best_rec_s_current_epoch, best_accu_s_current_epoch, best_pre_s_current_epoch, best_f1_s_current_epoch, best_fpr_s_current_epoch))
                print('Iter=%d, Metrics for the %s dataset: Recall: %4f, accu: %4f, Precision: %4f, F1: %4f, FPR: %4f'
                    % (itr, args.target_dataset_name, best_rec_t_current_epoch, best_accu_t_current_epoch, best_pre_t_current_epoch, best_f1_t_current_epoch, best_fpr_t_current_epoch))
                
                print('Iter=%d, Metrics for the %s dataset: Recall: %4f, accu: %4f, Precision: %4f, F1: %4f, FPR: %4f'
                    % (itr, args.target_2_dataset_name, best_rec_t2_current_epoch, best_accu_t2_current_epoch, best_pre_t2_current_epoch, best_f1_t2_current_epoch, best_fpr_t2_current_epoch))
                
                print('Corresponding model was save in ' + args.model_root + '/model_epoch_best.pth \n', flush=True)

        if best_f1_t_current_epoch > best_f1_t:
            best_f1_t = best_f1_t_current_epoch
            best_pre_t = best_pre_t_current_epoch
            best_rec_t = best_rec_t_current_epoch
            best_accu_t = best_accu_t_current_epoch

            best_f1_s = best_f1_s_current_epoch
            best_pre_s = best_pre_s_current_epoch
            best_rec_s = best_rec_s_current_epoch
            best_accu_s = best_accu_s_current_epoch
            
        if best_f1_t2_current_epoch > best_f1_t2:
            best_f1_t2 = best_f1_t2_current_epoch
            best_pre_t2 = best_pre_t2_current_epoch
            best_rec_t2 = best_rec_t2_current_epoch
            best_accu_t2 = best_accu_t2_current_epoch


        print('============Iter %s Summary ============= \n' % (str(itr+1)))
        print('Metrics for the %s dataset: Recall: %4f, accu: %4f, Precision: %4f, F1: %4f'
            % (args.source_dataset_name, best_rec_s, best_accu_s, best_pre_s, best_f1_s))
        print('Metrics for the %s dataset: Recall: %4f, accu: %4f, Precision: %4f, F1: %4f'
            % (args.target_dataset_name, best_rec_t, best_accu_t, best_pre_t, best_f1_t))
        
        print('Metrics for the %s dataset: Recall: %4f, accu: %4f, Precision: %4f, F1: %4f'
            % (args.target_2_dataset_name, best_rec_t2, best_accu_t2, best_pre_t2, best_f1_t2))
        print('Corresponding model was save in ' + args.model_root + '/model_epoch_best.pth')
        print('====================================== \n')
    
        results['src_pre'].update(best_pre_s.item())
        results['src_rec'].update(best_rec_s.item())
        results['src_f1'].update(best_f1_s.item())
        results['src_accu'].update(best_accu_s.item())
        
        results['tag_pre'].update(best_pre_t.item())
        results['tag_rec'].update(best_rec_t.item())
        results['tag_f1'].update(best_f1_t.item())
        results['tag_accu'].update(best_accu_t.item())

        results['tag_2_pre'].update(best_pre_t2.item())
        results['tag_2_rec'].update(best_rec_t2.item())
        results['tag_2_f1'].update(best_f1_t2.item())
        results['tag_2_accu'].update(best_accu_t2.item())
    
    print('============Train Summary (Average) ============= \n')
    print('Metrics for the %s dataset: Recall: %4f, accu: %4f, Precision: %4f, F1: %4f'
        % (args.source_dataset_name, results['src_rec'].avg, results['src_accu'].avg, results['src_pre'].avg, results['src_f1'].avg))
    print('Metrics for the %s dataset: Recall: %4f, accu: %4f, Precision: %4f, F1: %4f'
        % (args.target_dataset_name, results['tag_rec'].avg, results['tag_accu'].avg, results['tag_pre'].avg, results['tag_f1'].avg))
    print('Metrics for the %s dataset: Recall: %4f, accu: %4f, Precision: %4f, F1: %4f'
        % (args.target_2_dataset_name, results['tag_2_rec'].avg, results['tag_2_accu'].avg, results['tag_2_pre'].avg, results['tag_2_f1'].avg))
    print('====================================== \n')

if __name__ == "__main__":
    run_started = datetime.today().strftime('%d-%m-%y_%H%M')  # start time to create unique experiment name
    parser = argparse.ArgumentParser(description='GMBFD Training')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--iterations', default=3, type=int, help='number of total iterations to run')
    parser.add_argument('--n_epoch', default=30, type=int, help='number of total epochs to run')  # 1024
    parser.add_argument('--start_estimate_pl', default=15, type=int, help='start epoch of estimate pseudo label)')  # 30
    parser.add_argument('--batch_size', default=512, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate, default 0.03')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--seed', type=int, default=3047, help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--str_len', default=22, type=int)
    parser.add_argument('--class-blnc', default=7, type=int, help='total number of class balanced iterations')  # 10
    parser.add_argument('--tau-p', default=0.75, type=float,
                        help='confidece threshold for positive pseudo-labels, default 0.70')
    parser.add_argument('--tau-n', default=0.05, type=float,
                        help='confidece threshold for negative pseudo-labels, default 0.05')
    parser.add_argument('--kappa-p', default=0.05, type=float,
                        help='uncertainty threshold for positive pseudo-labels, default 0.05')
    parser.add_argument('--kappa-n', default=0.005, type=float,
                        help='uncertainty threshold for negative pseudo-labels, default 0.005')
    parser.add_argument('--temp-nl', default=2.0, type=float,
                        help='temperature for generating negative pseduo-labels, default 2.0')
    parser.add_argument('--no-uncertainty', default=False, type=bool,
                        help='use uncertainty in the pesudo-label selection, default False')
    parser.add_argument('--where_GPU', default='sutd', type=str, help='[wislab, sutd, sc]')
    parser.add_argument('--device', default='cuda:0', type=str, help='[cuda:0, cpu]')
    parser.add_argument('--model_root', default='models', type=str, help='store checkpoint.')
    parser.add_argument('--gat_hidden_dim', default=16, type=int)
    parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--source_dataset_name', default='sc_1', type=str)
    parser.add_argument('--target_dataset_name', default='sc_7', type=str)
    parser.add_argument('--target_2_dataset_name', default='sc_7', type=str)
    parser.add_argument('-gamma_loss', default=0.05)
    parser.add_argument('--loss_alpha', default=0.2, type=float,
                        help='confidece threshold for positive pseudo-labels, default 0.70')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    import json
    print(json.dumps(vars(args), indent=4))
    print('pid: ', os.getpid(), flush=True)
    main(args)

