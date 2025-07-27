import os
import torch.backends.cudnn as cudnn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import torch
from utils_var import getBatchedEdgeMat
import warnings
warnings.filterwarnings("ignore")


def test(dataloader, device, src_ip_lst, dst_ip_lst, batch_size=512, my_net=None):

    model_root = 'models'
    cuda = True
    cudnn.benchmark = True
    alpha = 0

    """ test """
    test_accs = []
    test_pre = []
    test_recall = []
    test_f1 = []
    test_fpr = []

    if my_net is None:
        my_net = torch.load(os.path.join(model_root, 'model_epoch_current.pth'))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.to(device)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0


    while i < len_dataloader:
        # test model using target data
        data_target = next(data_target_iter)
        t_num, t_str, t_label, _, t_src_ip, t_des_ip = data_target
        edge_mat = getBatchedEdgeMat(batch_size, t_src_ip, t_des_ip).to(device)

        batch_size = len(t_label)

        if cuda:
            t_num = t_num.to(device)
            t_str = t_str.to(device)
            t_label = t_label.to(device)

        class_output, _, _, _  = my_net(input_data_str=t_str, input_data_num=t_num, alpha=alpha, edge_index=edge_mat)


        prediction = class_output.data.max(1, keepdim=True)[1].detach().cpu().numpy()
        pre_ = precision_score(t_label.detach().cpu().numpy(), prediction, average='macro')
        rec_ = recall_score(t_label.detach().cpu().numpy(), prediction, average='macro')
        f1_ = f1_score(t_label.detach().cpu().numpy(), prediction, average='macro')
        accu_ = accuracy_score(t_label.detach().cpu().numpy(), prediction)
        

        conf_matrix = confusion_matrix(t_label.detach().cpu().numpy(), prediction)
        TN, FP, FN, TP = conf_matrix.ravel()
        fpr = FP / (FP + TN) if (FP + TN) != 0 else 0.0

        test_accs.append(accu_)
        test_pre.append(pre_)
        test_recall.append(rec_)
        test_f1.append(f1_)
        test_fpr.append(fpr)


        i += 1



    accu_ = np.mean(test_accs)
    pre_ = np.mean(test_pre)
    rec_ = np.mean(test_recall)
    f1_ = np.mean(test_f1)
    fpr_ = np.mean(test_fpr)
    return accu_, pre_, rec_, f1_, fpr_
