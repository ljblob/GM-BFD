import torch.nn as nn
from functions import ReverseLayerF
import torch, sys
from GAT import GAT
import numpy as np
from torch_geometric.nn import SAGEConv, GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn.models import GIN
from GAT import GATBatch

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='add', normalize=True))
        # self.conv = SAGEConv(in_channels, hidden_channels, aggr=aggregator, normalize=True)
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='add', normalize=True))
        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(p=0.15)
    def forward(self, x, edge_index):
        feature = x
        for conv in self.convs:
            aggre = conv(feature, edge_index)
            feature = self.dropout(aggre)
            feature = self.elu(feature)
        return feature

class GMBFD(nn.Module):

    def __init__(self, input_dim, cls_num, device, batch_sz, gat_hidden_dim, num_head):
        super(GMBFD, self).__init__()

        self.device = device
        self.batch_sz = batch_sz
        self.gat_hidden_dim = gat_hidden_dim
        self.num_head = num_head
        self.input_dim = input_dim
        self.classes = cls_num
        
        # GraphSAGE
        self.num_sage_layer = 1
        self.sage1 = GraphSAGE(in_channels=self.num_head*self.gat_hidden_dim, hidden_channels=self.num_head*self.gat_hidden_dim, num_layers=self.num_sage_layer)
        self.sage2 = GraphSAGE(in_channels=self.num_head*self.gat_hidden_dim, hidden_channels=self.num_head*self.gat_hidden_dim, num_layers=self.num_sage_layer)
        self.sage_ouput = torch.nn.Linear(in_features=self.num_head*self.gat_hidden_dim, out_features=self.num_head*self.gat_hidden_dim)
        
        self.leakyReLU = nn.LeakyReLU()  # using 0.2 as in the paper, no need to expose every setting
        self.param1 = nn.Parameter(torch.randn(1))
        self.param2 = nn.Parameter(torch.randn(1))
        self.softmax = torch.nn.Softmax(dim=0)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.num_head*self.gat_hidden_dim, 100))  # (16*24+16, 100)  (28*26+16, 100)
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, cls_num))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.num_head*self.gat_hidden_dim, 100))  #  (28*26+16, 100)
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        self.emb2 = nn.Embedding(122248, 16)  #  79928  93239  120095 122248
        

        self.conv = torch.nn.Sequential()
        self.conv.add_module('conv1', nn.Conv1d(self.gat_hidden_dim, self.gat_hidden_dim, 3, stride=1, padding=1))
        self.conv.add_module('conv1_bn1', nn.BatchNorm1d(self.gat_hidden_dim))
        self.conv.add_module('conv1_relu1', nn.ReLU(True))
        # self.conv.add_module('conv1_drop1', nn.Dropout1d(p=0.5))
        self.conv.add_module('conv2', nn.Conv1d(self.gat_hidden_dim, self.gat_hidden_dim, 3, stride=1, padding=1))
        self.conv.add_module('conv2_bn2', nn.BatchNorm1d(self.gat_hidden_dim))
        self.conv.add_module('conv2_relu2', nn.ReLU(True))
        # self.conv.add_module('conv2_drop1', nn.Dropout1d(p=0.6))

        self.gat = GATBatch(num_node_features=self.gat_hidden_dim, hidden_channels=self.gat_hidden_dim, num_heads=self.num_head, dropout=0.2).to(self.device)

        # local subdomain
        self.dcis = nn.Sequential()
        self.dci = {}
        for i in range(cls_num):   
            self.dci[i] = nn.Sequential()
            self.dci[i].add_module('fc1', nn.Linear(self.num_head*self.gat_hidden_dim, 256))
            self.dci[i].add_module('relu1', nn.ReLU(True))
            self.dci[i].add_module('dpt1', nn.Dropout())
            self.dci[i].add_module('fc2', nn.Linear(256, 256))
            self.dci[i].add_module('relu2', nn.ReLU(True))
            self.dci[i].add_module('dpt2', nn.Dropout())
            self.dci[i].add_module('fc3', nn.Linear(256, 2))
            self.dci[i].add_module('log_softmax', nn.LogSoftmax(dim=1))
            self.dcis.add_module('dci_'+str(i), self.dci[i])


    def forward(self, input_data_str, input_data_num, alpha, edge_index):  # input_data:[128,1,28,28] alpha:0.0
        input_emb = self.emb2(input_data_str)  # [b, 22] --> [b, 22, 16]
        input_data_num_expand = input_data_num.unsqueeze(2).expand(-1, -1, input_emb.shape[2])   # [b, 4] --> [b, 4, 28]
        input = torch.cat([input_emb, input_data_num_expand], dim=1)  # [b, 22, 16],[b, 4, 16]-->[b, 26, 16]

        # conv input: [batch, channels, seq_len], conv output: [batch, channels, seq_len]
        conv_out = self.conv(input.permute(0, 2, 1))  # [b, 26, 16]-[b, 16, 26] conv_out: [b, 16, 26]
        
        conv_out = conv_out.permute(0, 2, 1)

        # GAT
        self.num_nodes = conv_out.shape[1]
        graph_loader = self.load_graph_data(conv_out)

        for step, data in enumerate(graph_loader):
            feature_gat = self.gat(data.x, data.edge_index, data.batch)  # [b, 64] 64=16*4=feature_dim*head_num

        feature_out = feature_gat.clone()
        ###  GraphSAGE
        sage_feature1 = self.sage1(x=feature_gat, edge_index=edge_index)
        # sage_feature2 = self.sage2(x=sage_feature1, edge_index=edge_index)
        sage_output_ft = self.sage_ouput(sage_feature1)
        sage_output_ft = self.leakyReLU(sage_output_ft)

        feature_gat = sage_output_ft.clone()
        
        # RevGrad
        reverse_feature = ReverseLayerF.apply(feature_gat, alpha)  # [b,4*16]

        class_output = self.class_classifier(feature_gat)  # category classifier
        global_domain_output = self.domain_classifier(reverse_feature)  # global doimain classifier

        # local subdomain classifier
        local_domain_out = []

        # p*feature-> classifier_i ->loss_i
        for i in range(self.classes):  # self.classes:65
            prob = class_output[:, i].reshape((class_output.shape[0], 1))  
            fs = prob * reverse_feature  # [b,1]*[b,64]-->[b,64]

            out_i = self.dcis[i](fs)  # local domain discriminator  # [b,64]-->[b,2]
            local_domain_out.append(out_i)

        return class_output, global_domain_output, local_domain_out, feature_out.detach()    # [64,2],  [64,2], [64,2]


    def build_edge_index(self, adjacency_list_dict, num_of_nodes, add_self_edges=True):
        source_nodes_ids, target_nodes_ids = [], []
        seen_edges = set()

        for src_node, neighboring_nodes in adjacency_list_dict.items():
            for trg_node in neighboring_nodes:
                # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
                if (src_node, trg_node) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                    source_nodes_ids.append(src_node)
                    target_nodes_ids.append(trg_node)

                    seen_edges.add((src_node, trg_node))

        if add_self_edges:
            source_nodes_ids.extend(np.arange(num_of_nodes))
            target_nodes_ids.extend(np.arange(num_of_nodes))

        # shape = (2, E), where E is the number of edges in the graph
        edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

        return edge_index


    def load_graph_data(self, conv_out):
        '''for batch graph training'''
        adjacency_list = {node: [i for i in range(0, self.num_nodes) if i != node] for node in range(0, self.num_nodes)}
        topology = self.build_edge_index(adjacency_list, self.num_nodes, add_self_edges=True)  # (2, 25) shape = (2, E), where E is the number of edges, and 2 for source and target nodes.
        topology = torch.tensor(topology, dtype=torch.long, device=self.device)  # [2,676]

        all_data = []
        for i in range(conv_out.shape[0]):
            x_i = conv_out[i]
            edge_index_i = topology.clone()
            data = Data(x=x_i, edge_index=edge_index_i)
            all_data.append(data)

        graph_loader = DataLoader(all_data, batch_size=conv_out.shape[0], shuffle=False)

        return graph_loader



