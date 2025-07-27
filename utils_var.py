import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import yaml
from collections import Counter
import re
import os,sys
import scipy.sparse as sp


def getBatchedEdgeMat(batch_size, src_ip_lst, dst_ip_lst):
    edge_mat = np.zeros(shape=(batch_size, batch_size))
    for idx in range(batch_size):
        rowidx = idx
        # (i_src=j_src, i_dst=j_des)=4; 
        # (i_src=j_src, i_dst!=j_src)=3
        # (i_src!=j_src, i_dst=j_dst)=2; 
        # (i_src!=j_src, i_dst=j_src)=1
        # (i_src!=j_src, i_dst!=j_src)=0
        for colidx in range(rowidx+1, batch_size):
            weight = 0.
            if src_ip_lst[rowidx] == src_ip_lst[colidx]:
                if dst_ip_lst[rowidx] == dst_ip_lst[colidx]:
                    weight = 4
                else:
                    weight = 3
            else:
                if dst_ip_lst[rowidx] == dst_ip_lst[colidx]:
                    weight = 2
                else:
                    if (dst_ip_lst[rowidx] == src_ip_lst[colidx]) or (dst_ip_lst[colidx] == src_ip_lst[rowidx]):
                        weight = 1
            edge_mat[rowidx, colidx] = weight
            edge_mat[colidx, rowidx] = weight
        # add self-loops
        # edge_mat[rowidx, rowidx] = 4
    adj = sp.coo_matrix(edge_mat)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj.shape
    adj_tor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    adj_sp = adj_tor.coalesce()
    return adj_sp


def label2tensor(label):
    label_ = label.values
    label_tensor = torch.LongTensor(label_)
    return label_tensor


def normalize_num_data(data_df, num_col):
    num_data = data_df[num_col].copy()
    scaler = StandardScaler()  
    num_data_norm = scaler.fit_transform(num_data)
    num_data_norm_tensor = torch.FloatTensor(num_data_norm)
    return num_data_norm_tensor

class GetLoader(Dataset):
    def __init__(self, num_feature1, str_feature2, label, pseudo_idx=None, pseudo_target=None, ip_lst=None):
        super(Dataset, self).__init__()
        self.num_feature1 = num_feature1
        self.str_feature2 = str_feature2
        self.orig_indices = list(range(len(label)))
        self.label = label
        self.ip_lst = ip_lst

    def __getitem__(self, index):
        num_feature = self.num_feature1[index]
        str_feature = self.str_feature2[index]
        label = self.label[index]
        orig_index = self.orig_indices[index]
        src_ip = self.ip_lst[0][index]
        des_ip = self.ip_lst[1][index]
        return num_feature, str_feature, label, orig_index, src_ip, des_ip

    def __len__(self):
        return len(self.num_feature1)


def get_DataLoader(num_data_norm_tensor, input_ids, label, ip_lst=None, batchsize=10, shuffle=False, dropLast=True):
    dataset = GetLoader(num_feature1=num_data_norm_tensor, str_feature2=input_ids, label=label, ip_lst=ip_lst)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, drop_last=dropLast)
    return dataset, dataloader

def generate_mapping_dict(dataframe, column_name):
    df = dataframe.copy()
    unique_values = df[[column_name]].drop_duplicates()
    val_range = list(range(1, len(unique_values) + 1))
    unique_values['value'] = val_range
    mapping_dict = dict(zip(unique_values[column_name], unique_values['value']))

    return mapping_dict


def process_str(str_data):
    str_data['Dir'] = str_data['Dir'].str.replace(' ', '')
    Proto_dict = generate_mapping_dict(str_data, 'Proto')
    str_data['Proto'] = str_data['Proto'].replace(Proto_dict)

    Direction_dict = generate_mapping_dict(str_data, 'Dir')
    str_data['Dir'] = str_data['Dir'].replace(Direction_dict)

    State_dict = generate_mapping_dict(str_data, 'State')
    str_data['State'] = str_data['State'].replace(State_dict)

    sTos_dict = generate_mapping_dict(str_data, 'sTos')
    str_data['sTos'] = str_data['sTos'].replace(sTos_dict)

    dTos_dict = generate_mapping_dict(str_data, 'dTos')
    str_data['dTos'] = str_data['dTos'].replace(dTos_dict)

    str_data['Sport'] = str_data['Sport'].fillna(1e-6)
    Sport_dict = generate_mapping_dict(str_data, 'Sport')
    str_data['Sport'] = str_data['Sport'].map(Sport_dict)

    str_data['Dport'] = str_data['Dport'].fillna(1e-6)
    Dport_dict = generate_mapping_dict(str_data, 'Dport')
    str_data['Dport'] = str_data['Dport'].map(Dport_dict)

    Label = str_data[['Label']].drop_duplicates()

    dict_list = [Proto_dict, Direction_dict, State_dict]

    with open('data.yaml', 'w') as file:
        yaml.dump(dict_list, file)

    return str_data


def encode_str(netflow, vocab, str_col, num_to_encode=3):

    str_data = split_word(netflow, str_col) 
    vocab_dict = create_vocab_dict(vocab=vocab, num_to_encode=num_to_encode)
    max_length, input_ids, attention_mask = get_token(str_data['text'], vocab_dict)
    return max_length, input_ids, attention_mask


def get_token(str_data, vocab_dict):

    str_data_text = str_data.values.tolist()  
    str_data_text_mapped = [[vocab_dict.get(token, token) for token in item.split()] for item in
                            str_data_text] 
    str_data_text_mapped_ = [[1] + sublist + [2] for sublist in str_data_text_mapped]
    max_length = 22
    input_ids = [fill_padding(sublist, max_length) for sublist in str_data_text_mapped_]  
    attention_mask = [[1] * len(sub_list) for sub_list in str_data_text_mapped_]
    attention_mask = [fill_padding(sub_list, max_length) for sub_list in attention_mask]  

    return max_length, input_ids, attention_mask


def fill_padding(data, max_len):
    if len(data) < max_len:
        pad_len = max_len-len(data)
        padding = [0 for _ in range(pad_len)]
        data = torch.tensor(data+padding)
    else:
        data = torch.tensor(data[:max_len])
    return data


def create_vocab_dict(vocab, num_to_encode=3):
    vocab_dict = {}
    for index, line in enumerate(vocab):
        item = line.strip("\n").split()[0] if line.strip() else line.strip("\n")
        vocab_dict[item] = index + num_to_encode  # 从3开始编码
        add_kv = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2}
        vocab_dict.update(add_kv)
    return vocab_dict


def split_word(netflow, str_col):

    str_data = netflow[str_col].astype(str) 
    str_data = str_data.applymap(lambda x: ' '.join(x.split()) if isinstance(x, str) else x) 
    sep_pattern_time = r'[/:.]'
    sep_pattern_proto = r'[-/]'
    str_data['Proto'] = str_data['Proto'].apply(lambda x: ' '.join(re.split(sep_pattern_proto, x)))
    str_data['text'] = str_data.apply(lambda x: ' '.join(x), axis=1)
    return str_data


def preprocess_data(data, vocab, str_col):
    max_length, input_ids, _ = encode_str(data, vocab, str_col, num_to_encode=3)
    label = label2tensor(data['Label'])
    num_col = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']  
    num_norm_tensor = normalize_num_data(data, num_col)
    return input_ids, num_norm_tensor, label


def load_files(dataset_root_path, vocab_root_path, source_domain_fg, target_domain_fg, where_GPU):

    file_name_dict = {'1': 'capture20110810_preprocessed_1103_label.binetflow',
                        '2': 'capture20110811_preprocessed_0204_label.binetflow',
                        '3': 'capture20110812_preprocessed_1125_label.binetflow',
                        '4': 'capture20110815_preprocessed_0204_label.binetflow',
                        '5': 'capture20110815-2_preprocessed_0204_label.binetflow',
                        '6': 'capture20110816_preprocessed_0204_label.binetflow',
                        '7': 'capture20110816-2_preprocessed_0204_label.binetflow',
                        '8': 'capture20110816-3_preprocessed_240613_label.binetflow',
                        '9': 'capture20110817_preprocessed_0204_label.binetflow',
                        '10':'capture20110818_preprocessed_0204_label.binetflow',
                        '11': 'capture20110818-2_preprocessed_0204_label.binetflow',
                        '12': 'capture20110819_preprocessed_0204_label.binetflow',
                        '13': 'capture20110815-3_preprocessed_0204_label.binetflow'}

    vocab_name_dict = {'1': 'encrypted_vocab_exclude_time_1_1121.txt',
                        '2': 'encrypted_vocab_exclude_time_2_1121.txt',
                        '3': 'encrypted_vocab_exclude_time_3_1125.txt',
                        '4': 'encrypted_vocab_exclude_time_4_0205.txt',
                        '5': 'encrypted_vocab_exclude_time_5_0205.txt',
                        '6': 'encrypted_vocab_exclude_time_6_0205.txt',
                        '7': 'encrypted_vocab_exclude_time_7_0205.txt',
                        '8': 'encrypted_vocab_exclude_time_8_0205.txt',
                        '9': 'encrypted_vocab_exclude_time_9_0216.txt',
                        '10': 'encrypted_vocab_exclude_time_10_0216.txt',
                        '11': 'encrypted_vocab_exclude_time_11_0205.txt',
                        '12': 'encrypted_vocab_exclude_time_12_0205.txt',
                        '13': 'encrypted_vocab_exclude_time_13_0205.txt'}


    source_file_name = file_name_dict[source_domain_fg]
    target_file_name = file_name_dict[target_domain_fg]

    source_file_path = os.path.join(dataset_root_path, source_domain_fg, source_file_name)
    target_file_path = os.path.join(dataset_root_path, target_domain_fg, target_file_name)

    src_file = pd.read_csv(source_file_path)
    tag_file = pd.read_csv(target_file_path)

    # if where_GPU == 'sc' and source_domain_fg == '1':
    #     src_file['Label'] = src_file['Label'] - 1

    # remove background netflow
    src_data = src_file[(src_file['Label'] == 1) | (src_file['Label'] == 2)]
    tag_data = tag_file[(tag_file['Label'] == 1) | (tag_file['Label'] == 2)]
    src_data['Label'] = src_data['Label'] - 1
    tag_data['Label'] = tag_data['Label'] - 1

    print(Counter(src_data['Label']))
    print(Counter(tag_data['Label']))

    # load vocab files
    source_vocab_name = vocab_name_dict[source_domain_fg]
    target_vocab_name = vocab_name_dict[target_domain_fg]

    source_vocab_path = os.path.join(vocab_root_path, source_vocab_name)
    target_vocab_path = os.path.join(vocab_root_path, target_vocab_name)

    src_vocab = np.loadtxt(source_vocab_path, dtype=str).tolist()
    trg_vocab = np.loadtxt(target_vocab_path, dtype=str).tolist()

    src_vocab.extend(trg_vocab)
    vocab = list(set(src_vocab))

    return src_data, tag_data, vocab


def print_learning_rate(optimizer):
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k == 'params':
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        print(outputs)


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def load_files_multiTarget(dataset_root_path, vocab_root_path, source_domain_fg, target_domain_fg,target_2_domain_fg, where_GPU):


    file_name_dict = {'1': 'capture20110810_preprocessed_1103_label.binetflow',
                        '2': 'capture20110811_preprocessed_0204_label.binetflow',
                        '3': 'capture20110812_preprocessed_1125_label.binetflow',
                        '4': 'capture20110815_preprocessed_0204_label.binetflow',
                        '5': 'capture20110815-2_preprocessed_0204_label.binetflow',
                        '6': 'capture20110816_preprocessed_0204_label.binetflow',
                        '7': 'capture20110816-2_preprocessed_0204_label.binetflow',
                        '8': 'capture20110816-3_preprocessed_240613_label.binetflow',
                        '9': 'capture20110817_preprocessed_0204_label.binetflow',
                        '10':'capture20110818_preprocessed_0204_label.binetflow',
                        '11': 'capture20110818-2_preprocessed_0204_label.binetflow',
                        '12': 'capture20110819_preprocessed_0204_label.binetflow',
                        '13': 'capture20110815-3_preprocessed_0204_label.binetflow'}

    vocab_name_dict = {'1': 'encrypted_vocab_exclude_time_1_1121.txt',
                        '2': 'encrypted_vocab_exclude_time_2_1121.txt',
                        '3': 'encrypted_vocab_exclude_time_3_1125.txt',
                        '4': 'encrypted_vocab_exclude_time_4_0205.txt',
                        '5': 'encrypted_vocab_exclude_time_5_0205.txt',
                        '6': 'encrypted_vocab_exclude_time_6_0205.txt',
                        '7': 'encrypted_vocab_exclude_time_7_0205.txt',
                        '8': 'encrypted_vocab_exclude_time_8_0205.txt',
                        '9': 'encrypted_vocab_exclude_time_9_0216.txt',
                        '10': 'encrypted_vocab_exclude_time_10_0216.txt',
                        '11': 'encrypted_vocab_exclude_time_11_0205.txt',
                        '12': 'encrypted_vocab_exclude_time_12_0205.txt',
                        '13': 'encrypted_vocab_exclude_time_13_0205.txt'}



    source_file_name = file_name_dict[source_domain_fg]
    target_file_name = file_name_dict[target_domain_fg]
    target_2_file_name = file_name_dict[target_2_domain_fg]

    source_file_path = os.path.join(dataset_root_path, source_domain_fg, source_file_name)
    target_file_path = os.path.join(dataset_root_path, target_domain_fg, target_file_name)
    target_2_file_path = os.path.join(dataset_root_path, target_2_domain_fg, target_2_file_name)

    src_file = pd.read_csv(source_file_path)
    tag_file = pd.read_csv(target_file_path)
    tag_2_file = pd.read_csv(target_2_file_path)

    # if where_GPU == 'sc' and source_domain_fg == '1':
    #     src_file['Label'] = src_file['Label'] - 1

    # remove background netflow
    src_data = src_file[(src_file['Label'] == 1) | (src_file['Label'] == 2)]
    tag_data = tag_file[(tag_file['Label'] == 1) | (tag_file['Label'] == 2)]
    tag_2_data = tag_2_file[(tag_2_file['Label'] == 1) | (tag_2_file['Label'] == 2)]
    src_data['Label'] = src_data['Label'] - 1
    tag_data['Label'] = tag_data['Label'] - 1
    tag_2_data['Label'] = tag_2_data['Label'] - 1

    print('Source domain (scenario={source_domain_fg}) label distribution:', Counter(src_data['Label']))
    print('Target domain 1 (scenario={target_domain_fg}) label distribution:', Counter(tag_data['Label']))
    print('Target domain 2 (scenario={target_2_domain_fg}) label distribution:', Counter(tag_2_data['Label']))
    # print(Counter(tag_data['Label']))
    # print(Counter(tag_2_data['Label']))

    # load vocab files
    source_vocab_name = vocab_name_dict[source_domain_fg]
    target_vocab_name = vocab_name_dict[target_domain_fg]
    target_2_vocab_name = vocab_name_dict[target_2_domain_fg]

    source_vocab_path = os.path.join(vocab_root_path, source_vocab_name)
    target_vocab_path = os.path.join(vocab_root_path, target_vocab_name)
    target_2_vocab_path = os.path.join(vocab_root_path, target_2_vocab_name)

    src_vocab = np.loadtxt(source_vocab_path, dtype=str).tolist()
    trg_vocab = np.loadtxt(target_vocab_path, dtype=str).tolist()
    trg_2_vocab = np.loadtxt(target_2_vocab_path, dtype=str).tolist()

    src_vocab.extend(trg_vocab)
    src_vocab.extend(trg_2_vocab)
    vocab = list(set(src_vocab))

    return src_data, tag_data, tag_2_data, vocab


