import sys
import numpy as np
import random
sys.path.append('..')
import nrekit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import sklearn.metrics 
import copy

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx



class EnhancedGNN(nn.Module):
    """
    Graph Neural Network for refining RSN-based node features.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, use_attention=True):
        super(EnhancedGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Add GNN layers (GCN or GAT)
        for i in range(num_layers):
            if use_attention:
                layer = GATConv(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim if i < num_layers - 1 else output_dim,
                    heads=1 if i == num_layers - 1 else 8,
                    concat=(i < num_layers - 1),
                    dropout=0.5,
                )
            else:
                layer = GCNConv(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim if i < num_layers - 1 else output_dim,
                )
            self.layers.append(layer)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through GNN layers.
        :param x: Node features.
        :param edge_index: Edge connections.
        :param edge_weight: Edge weights for GCN (optional).
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GCNConv):  # GCN
                x = F.relu(layer(x, edge_index, edge_weight))
            else:  # GAT
                x = F.elu(layer(x, edge_index))
            if i < self.num_layers - 1:  # Apply dropout between layers
                x = self.dropout(x)
        return x


class Siamese(nn.Module):

    def __init__(self, sentence_encoder, hidden_size=230, drop_rate=0.5, pre_rep=None, euc=True):
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder # Should be different from main sentence encoder
        self.hidden_size = hidden_size
        # self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        # self.fc2 = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size, 1)
        self.cost = nn.BCELoss(reduction="none")
        self.drop = nn.Dropout(drop_rate)
        self._accuracy = 0.0
        self.pre_rep = pre_rep
        self.euc = euc

    def forward(self, data, num_size, num_class, threshold=0.5):
        x = self.sentence_encoder(data).contiguous().view(num_class, num_size, -1)
        x1 = x[:, :num_size//2].contiguous().view(-1, self.hidden_size)
        x2 = x[:, num_size//2:].contiguous().view(-1, self.hidden_size)
        y1 = x[:num_class//2,:].contiguous().view(-1, self.hidden_size)
        y2 = x[num_class//2:,:].contiguous().view(-1, self.hidden_size)
        # y1 = x[0].contiguous().unsqueeze(0).expand(x.size(0) - 1, -1, -1).contiguous().view(-1, self.hidden_size)
        # y2 = x[1:].contiguous().view(-1, self.hidden_size)

        label = torch.zeros((x1.size(0) + y1.size(0))).long().cuda()
        label[:x1.size(0)] = 1
        z1 = torch.cat([x1, y1], 0)
        z2 = torch.cat([x2, y2], 0)

        if self.euc:
            dis = torch.pow(z1 - z2, 2)
            dis = self.drop(dis)
            score = torch.sigmoid(self.fc(dis).squeeze())
        else:
            z = z1 * z2
            z = self.drop(z)
            z = self.fc(z).squeeze()
            # z = torch.cat([z1, z2], -1)
            # z = F.relu(self.fc1(z))
            # z = self.fc2(z).squeeze()
            score = torch.sigmoid(z)

        self._loss = self.cost(score, label.float()).mean()
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        self._accuracy = torch.mean((pred == label).type(torch.FloatTensor))
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        self._prec = float(np.logical_and(pred == 1, label == 1).sum()) / float((pred == 1).sum() + 1)
        self._recall = float(np.logical_and(pred == 1, label == 1).sum()) / float((label == 1).sum() + 1)

    def encode(self, dataset, batch_size=0): 
        self.sentence_encoder.eval()
        with torch.no_grad():
            if self.pre_rep is not None:
                return self.pre_rep[dataset['id'].view(-1)] 

            if batch_size == 0:
                x = self.sentence_encoder(dataset)
            else:
                total_length = dataset['word'].size(0)
                max_iter = total_length // batch_size
                if total_length % batch_size != 0:
                    max_iter += 1
                x = []
                for it in range(max_iter):
                    scope = list(range(batch_size * it, min(batch_size * (it + 1), total_length)))
                    with torch.no_grad():
                        _ = {'word': dataset['word'][scope], 'mask': dataset['mask'][scope]}
                        if 'pos1' in dataset:
                            _['pos1'] = dataset['pos1'][scope]
                            _['pos2'] = dataset['pos2'][scope]
                        _x = self.sentence_encoder(_)
                    x.append(_x.detach())
                x = torch.cat(x, 0)
            return x

    def forward_infer(self, x, y, threshold=0.5, batch_size=0):
        x = self.encode(x, batch_size=batch_size)
        support_size = x.size(0)
        y = self.encode(y, batch_size=batch_size)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

        if self.euc:
            dis = torch.pow(x - y, 2)
            score = torch.sigmoid(self.fc(dis).squeeze(-1)).mean(0)
        else:
            z = x * y
            z = self.fc(z).squeeze(-1)
            score = torch.sigmoid(z).mean(0)

        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        pred = pred.view(support_size, -1).sum(0)
        pred[pred < 1] = 0
        pred[pred > 0] = 1
        return pred

    def forward_infer_sort(self, x, y, batch_size=0):
        x = self.encode(x, batch_size=batch_size)
        support_size = x.size(0)
        y = self.encode(y, batch_size=batch_size)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

        if self.euc:
            dis = torch.pow(x - y, 2)
            score = torch.sigmoid(self.fc(dis).squeeze(-1)).mean(0)
        else:
            z = x * y
            z = self.fc(z).squeeze(-1)
            score = torch.sigmoid(z).mean(0)

        pred = []
        for i in range(score.size(0)):
            pred.append((score[i], i))
        pred.sort(key=lambda x: x[0], reverse=True)
        return pred

class Snowball(nrekit.framework.Model):
    
    def __init__(self, sentence_encoder, base_class, siamese_model, hidden_size=230, drop_rate=0.5, weight_table=None, pre_rep=None, neg_loader=None, args=None):
        nrekit.framework.Model.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.base_class = base_class
        self.fc = nn.Linear(hidden_size, base_class)
        self.drop = nn.Dropout(drop_rate)
        self.siamese_model = siamese_model
        # self.cost = nn.BCEWithLogitsLoss()
        self.cost = nn.BCELoss(reduction="none")
        # self.cost = nn.CrossEntropyLoss()
        self.weight_table = weight_table
        
        self.args = args

        self.pre_rep = pre_rep
        self.neg_loader = neg_loader

        # Initialize lists to store metrics
        self.similarity_scores = []  # Initialize similarity scores list
        self.accuracy_history = []    # Initialize accuracy history list

    # def __loss__(self, logits, label):
    #     onehot_label = torch.zeros(logits.size()).cuda()
    #     onehot_label.scatter_(1, label.view(-1, 1), 1)
    #     return self.cost(logits, onehot_label)

    # def __loss__(self, logits, label):
    #     return self.cost(logits, label)
    
    def construct_graph(self, node_features, similarity_scores, threshold=0.5):
        """
        Constructs a graph using similarity scores as edge weights.
        :param node_features: RSN sentence embeddings (node features).
        :param similarity_scores: List of (score, idx) from RSN.
        :param threshold: Minimum similarity score to form an edge.
        :return: edge_index, edge_weight.
        """
        edge_index, edge_weights = [], []
        for score, (i, j) in similarity_scores:
            if score > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Bidirectional edges
                edge_weights.append(score)
                edge_weights.append(score)

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().cuda()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float).cuda()
        else:
            edge_index, edge_weights = None, None

        return edge_index, edge_weights

    def forward_base(self, data):
            batch_size = data['word'].size(0)
            x = self.sentence_encoder(data) # (batch_size, hidden_size)
            x = self.drop(x)
            x = self.fc(x) # (batch_size, base_class)

            x = torch.sigmoid(x)
            if self.weight_table is None:
                weight = 1.0
            else:
                weight = self.weight_table[data['label']].unsqueeze(1).expand(-1, self.base_class).contiguous().view(-1)
            label = torch.zeros((batch_size, self.base_class)).cuda()
            label.scatter_(1, data['label'].view(-1, 1), 1) # (batch_size, base_class)
            loss_array = self.__loss__(x, label)
            self._loss = ((label.view(-1) + 1.0 / self.base_class) * weight * loss_array).mean() * self.base_class
            # self._loss = self.__loss__(x, data['label'])
            
            _, pred = x.max(-1)
            self._accuracy = self.__accuracy__(pred, data['label'])
            self._pred = pred
    
    def forward_baseline(self, support_pos, query, threshold=0.5):
        '''
        baseline model
        support_pos: positive support set
        support_neg: negative support set
        query: query set
        threshold: ins whose prob > threshold are predicted as positive
        '''
        
        # train
        self._train_finetune_init()
        # support_rep = self.encode(support, self.args.infer_batch_size)
        support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
        # self._train_finetune(support_rep, support['label'])
        self._train_finetune(support_pos_rep)

        
        # test
        query_prob = self._infer(query, batch_size=self.args.infer_batch_size).cpu().detach().numpy()
        label = query['label'].cpu().detach().numpy()
        self._baseline_accuracy = float(np.logical_or(np.logical_and(query_prob > threshold, label == 1), np.logical_and(query_prob < threshold, label == 0)).sum()) / float(query_prob.shape[0])
        if (query_prob > threshold).sum() == 0:
            self._baseline_prec = 0
        else:        
            self._baseline_prec = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((query_prob > threshold).sum())
        self._baseline_recall = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((label == 1).sum())
        if self._baseline_prec + self._baseline_recall == 0:
            self._baseline_f1 = 0
        else:
            self._baseline_f1 = float(2.0 * self._baseline_prec * self._baseline_recall) / float(self._baseline_prec + self._baseline_recall)
        self._baseline_auc = sklearn.metrics.roc_auc_score(label, query_prob)
        if self.args.print_debug:
            print('')
            sys.stdout.write('[BASELINE EVAL] acc: {0:2.2f}%, prec: {1:2.2f}%, rec: {2:2.2f}%, f1: {3:1.3f}, auc: {4:1.3f}'.format( \
                self._baseline_accuracy * 100, self._baseline_prec * 100, self._baseline_recall * 100, self._baseline_f1, self._baseline_auc))
            print('')

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward_few_shot_baseline(self, support, query, label, B, N, K, Q):
        support_rep = self.encode(support, self.args.infer_batch_size)
        query_rep = self.encode(query, self.args.infer_batch_size)
        support_rep.view(B, N, K, -1)
        query_rep.view(B, N * Q, -1)
        
        NQ = N * Q
         
        # Prototypical Networks 
        proto = torch.mean(support_rep, 2) # Calculate prototype for each class
        logits = -self.__batch_dist__(proto, query)
        _, pred = torch.max(logits.view(-1, N), 1)

        self._accuracy = self.__accuracy__(pred.view(-1), label.view(-1))

        return logits, pred

#    def forward_few_shot(self, support, query, label, B, N, K, Q):
#        for b in range(B):
#            for n in range(N):
#                _forward_train(self, support_pos, None, query, distant, threshold=0.5):
#
#        '''
#        support_rep = self.encode(support, self.args.infer_batch_size)
#        query_rep = self.encode(query, self.args.infer_batch_size)
#        support_rep.view(B, N, K, -1)
#        query_rep.view(B, N * Q, -1)
#        '''
#        
#        proto = []
#        for b in range(B):
#            for N in range(N)
#        
#        NQ = N * Q
#         
#        # Prototypical Networks 
#        proto = torch.mean(support_rep, 2) # Calculate prototype for each class
#        logits = -self.__batch_dist__(proto, query)
#        _, pred = torch.max(logits.view(-1, N), 1)
#
#        self._accuracy = self.__accuracy__(pred.view(-1), label.view(-1))
#
#        return logits, pred

    def _train_finetune_init(self):
        # init variables and optimizer
        self.new_W = Variable(self.fc.weight.mean(0) / 1e3, requires_grad=True)
        self.new_bias = Variable(torch.zeros((1)), requires_grad=True)
        self.optimizer = optim.Adam([self.new_W, self.new_bias], self.args.finetune_lr, weight_decay=self.args.finetune_wd)
        self.new_W = self.new_W.cuda()
        self.new_bias = self.new_bias.cuda()
    
    def _train_finetune(self, data_repre, learning_rate=None, labels=None, edge_index=None, edge_weights=None, num_epochs=10, weight_decay=1e-5):
        """
        Fine-tune classifier with GNN integration while retaining original logic.
        :param data_repre: Sentence representation (encoder's output).
        :param labels: Ground-truth labels.
        :param edge_index: Graph connectivity.
        :param edge_weights: Edge weights from similarity scores.
        :param learning_rate: Optional learning rate for optimizer.
        :param num_epochs: Number of epochs for fine-tuning.
        :param weight_decay: Weight decay for optimizer.
        """
        self.train()

        # Optimizer setup
        optimizer = self.optimizer
        if learning_rate is not None:
            optimizer = optim.Adam([self.new_W, self.new_bias], learning_rate, weight_decay=weight_decay)

        # Hyperparameters
        max_epoch = num_epochs
        batch_size = self.args.finetune_batch_size

        # Apply dropout to the data representation
        data_repre = self.drop(data_repre)

        # Debugging and logging
        if self.args.print_debug:
            print('')
        
        loss_history = []  # To track loss for plotting

        # Training loop
        for epoch in range(max_epoch):
            max_iter = data_repre.size(0) // batch_size
            if data_repre.size(0) % batch_size != 0:
                max_iter += 1
            order = list(range(data_repre.size(0)))
            random.shuffle(order)

            for i in range(max_iter):
                # Batch preparation
                x = data_repre[order[i * batch_size: min((i + 1) * batch_size, data_repre.size(0))]]
                batch_label = torch.ones((x.size(0))).long().cuda()

                # Negative sampling
                neg_size = int(x.size(0) * 1)
                neg = self.neg_loader.next_batch(neg_size)
                neg = self.encode(neg, self.args.infer_batch_size)

                # Apply GNN to negative samples if graph exists
                if edge_index is not None and edge_weights is not None:
                    neg_enhanced = self.gnn(neg, edge_index, edge_weights)
                    neg = torch.cat((neg, neg_enhanced), dim=1)  # Concatenate negative features

                # Combine positive and negative samples
                x = torch.cat([x, neg], 0)
                batch_label = torch.cat([batch_label, torch.zeros((neg_size)).long().cuda()], 0)

                # Apply GNN to positive samples if graph exists
                if edge_index is not None and edge_weights is not None:
                    gnn_enhanced_rep = self.gnn(x, edge_index, edge_weights)
                    x = torch.cat((x, gnn_enhanced_rep), dim=1)  # Concatenate GNN-enhanced features

                # Forward pass
                x = torch.matmul(x, self.new_W) + self.new_bias  # (batch_size, 1)
                x = torch.sigmoid(x)

                # Compute loss with weighted negative samples
                weight = torch.ones(batch_label.size(0)).float().cuda()
                weight[batch_label == 0] = self.args.finetune_weight
                iter_loss = (self.__loss__(x, batch_label.float()) * weight).mean()

                # Optimization
                optimizer.zero_grad()
                iter_loss.backward(retain_graph=True)
                optimizer.step()
                
                 # Calculate accuracy
                accuracy = (x.round() == batch_label.float()).float().mean().item()
                self.accuracy_history.append(accuracy)  # Log accuracy


                # Logging for debugging
                if self.args.print_debug:
                    sys.stdout.write('[snowball finetune] epoch {0:4} iter {1:4} | loss: {2:2.6f}'.format(epoch, i, iter_loss) + '\r')
                    sys.stdout.flush()

                # Append loss for visualization
                loss_history.append(iter_loss.item())

        self.eval()

        # Plot loss history for presentation
        self.plot_loss_history(loss_history)

    def plot_loss_history(self, loss_history):
        """
        Plot loss history for fine-tuning.
        :param loss_history: List of loss values over training iterations.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(loss_history, label="Loss", color="red")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss Curve During Fine-Tuning")
        plt.legend()
        plt.show()

    def _add_ins_to_data(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst (list)
        dataset_dst: destination dataset
        dataset_src: source dataset
        ins_id: id of the instance
        '''
        dataset_dst['word'].append(dataset_src['word'][ins_id])
        if 'pos1' in dataset_src:
            dataset_dst['pos1'].append(dataset_src['pos1'][ins_id])
            dataset_dst['pos2'].append(dataset_src['pos2'][ins_id])
        dataset_dst['mask'].append(dataset_src['mask'][ins_id])
        if 'id' in dataset_dst and 'id' in dataset_src:
            dataset_dst['id'].append(dataset_src['id'][ins_id])
        if 'entpair' in dataset_dst and 'entpair' in dataset_src:
            dataset_dst['entpair'].append(dataset_src['entpair'][ins_id])
        if 'label' in dataset_dst and label is not None:
            dataset_dst['label'].append(label)

    def _add_ins_to_vdata(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst (variable)
        dataset_dst: destination dataset
        dataset_src: source dataset
        ins_id: id of the instance
        '''
        dataset_dst['word'] = torch.cat([dataset_dst['word'], dataset_src['word'][ins_id].unsqueeze(0)], 0)
        if 'pos1' in dataset_src:
            dataset_dst['pos1'] = torch.cat([dataset_dst['pos1'], dataset_src['pos1'][ins_id].unsqueeze(0)], 0)
            dataset_dst['pos2'] = torch.cat([dataset_dst['pos2'], dataset_src['pos2'][ins_id].unsqueeze(0)], 0)
        dataset_dst['mask'] = torch.cat([dataset_dst['mask'], dataset_src['mask'][ins_id].unsqueeze(0)], 0)
        if 'id' in dataset_dst and 'id' in dataset_src:
            dataset_dst['id'] = torch.cat([dataset_dst['id'], dataset_src['id'][ins_id].unsqueeze(0)], 0)
        if 'entpair' in dataset_dst and 'entpair' in dataset_src:
            dataset_dst['entpair'].append(dataset_src['entpair'][ins_id])
        if 'label' in dataset_dst and label is not None:
            dataset_dst['label'] = torch.cat([dataset_dst['label'], torch.ones((1)).long().cuda()], 0)

    def _dataset_stack_and_cuda(self, dataset):
        '''
        stack the dataset to torch.Tensor and use cuda mode
        dataset: target dataset
        '''
        if (len(dataset['word']) == 0):
            return
        dataset['word'] = torch.stack(dataset['word'], 0).cuda()
        if 'pos1' in dataset:
            dataset['pos1'] = torch.stack(dataset['pos1'], 0).cuda()
            dataset['pos2'] = torch.stack(dataset['pos2'], 0).cuda()
        dataset['mask'] = torch.stack(dataset['mask'], 0).cuda()
        dataset['id'] = torch.stack(dataset['id'], 0).cuda()

    def encode(self, dataset, batch_size=0):
        self.sentence_encoder.eval()
        with torch.no_grad():
            if self.pre_rep is not None:
                return self.pre_rep[dataset['id'].view(-1)]

            if batch_size == 0:
                x = self.sentence_encoder(dataset)
            else:
                total_length = dataset['word'].size(0)
                max_iter = total_length // batch_size
                if total_length % batch_size != 0:
                    max_iter += 1
                x = []
                for it in range(max_iter):
                    scope = list(range(batch_size * it, min(batch_size * (it + 1), total_length)))
                    with torch.no_grad():
                        _ = {'word': dataset['word'][scope], 'mask': dataset['mask'][scope]}
                        if 'pos1' in dataset:
                            _['pos1'] = dataset['pos1'][scope]
                            _['pos2'] = dataset['pos2'][scope]
                        _x = self.sentence_encoder(_)
                    x.append(_x.detach())
                x = torch.cat(x, 0)
            return x

    def _infer(self, dataset, batch_size=0):
        '''
        get prob output of the finetune network with the input dataset
        dataset: input dataset
        return: prob output of the finetune network
        '''
        x = self.encode(dataset, batch_size=batch_size) 
        x = torch.matmul(x, self.new_W) + self.new_bias # (batch_size, 1)
        x = torch.sigmoid(x)
        return x.view(-1)

    def _forward_train(self, support_pos, query, distant, threshold=0.5):
        """
        Snowball training process with GNN integration and graphical logging.
        :param support_pos: Positive support set (raw data).
        :param query: Query set.
        :param distant: Distant data loader.
        :param threshold: Instances with prob > threshold are classified as positive.
        """
        # Hyperparameters
        snowball_max_iter = self.args.snowball_max_iter
        candidate_num_class = 20
        candidate_num_ins_per_class = 100

        sort_num1 = self.args.phase1_add_num
        sort_num2 = self.args.phase2_add_num
        sort_threshold1 = self.args.phase1_siamese_th
        sort_threshold2 = self.args.phase2_siamese_th
        sort_ori_threshold = self.args.phase2_cl_th

        # Initialize support set representation
        self._train_finetune_init()
        support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
        self._train_finetune(support_pos_rep)  # Initial fine-tuning

        # Log initial embeddings
        self.plot_embeddings(support_pos_rep, support_pos['label'], title="Initial Embeddings")

        # Keep metrics for visualization
        self._metric = []

        # Tracking existing IDs
        exist_id = {support_pos['id'][i]: 1 for i in range(len(support_pos['id']))}

        # Begin Snowball iterations
        for snowball_iter in range(snowball_max_iter):
            print(f"=== Snowball Iteration {snowball_iter + 1}/{snowball_max_iter} ===")

            # Phase 1: Expand positive support set from distant dataset
            self._phase1_add_num = 0
            self._phase1_total = 0

            entpair_support = {}
            entpair_distant = {}

            # Group support instances by entity pairs
            for i in range(len(support_pos['id'])):
                entpair = support_pos['entpair'][i]
                if entpair not in entpair_support:
                    if 'pos1' in support_pos:
                        entpair_support[entpair] = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
                    else:
                        entpair_support[entpair] = {'word': [], 'mask': [], 'id': []}
                self._add_ins_to_data(entpair_support[entpair], support_pos, i)

            # Process distant instances with the same entity pairs
            for entpair in entpair_support:
                raw = distant.get_same_entpair_ins(entpair)
                if raw is None:
                    continue
                if 'pos1' in support_pos:
                    entpair_distant[entpair] = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': []}
                else:
                    entpair_distant[entpair] = {'word': [], 'mask': [], 'id': [], 'entpair': []}
                for i in range(raw['word'].size(0)):
                    if raw['id'][i] not in exist_id:
                        self._add_ins_to_data(entpair_distant[entpair], raw, i)

                # Convert data to tensors
                self._dataset_stack_and_cuda(entpair_support[entpair])
                self._dataset_stack_and_cuda(entpair_distant[entpair])
                if len(entpair_support[entpair]['word']) == 0 or len(entpair_distant[entpair]['word']) == 0:
                    continue

                # Use Siamese model to rank instances
                pick_or_not = self.siamese_model.forward_infer_sort(
                    entpair_support[entpair], entpair_distant[entpair], batch_size=self.args.infer_batch_size
                )

                # Construct graph from similarity scores
                edge_index, edge_weights = self.construct_graph(support_pos_rep, pick_or_not)

                # Visualize graph
                self.plot_graph(edge_index)

                # Fine-tune with GNN-enhanced features
                if edge_index is not None:
                    self._train_finetune(
                        support_pos_rep, labels=support_pos['label'], edge_index=edge_index, edge_weights=edge_weights
                    )

                # Expand support set with high-confidence instances
                for i in range(min(len(pick_or_not), sort_num1)):
                    if pick_or_not[i][0] > sort_threshold1:
                        iid = pick_or_not[i][1]
                        self._add_ins_to_vdata(support_pos, entpair_distant[entpair], iid, label=1)
                        exist_id[entpair_distant[entpair]['id'][iid]] = 1
                        self._phase1_add_num += 1
                self._phase1_total += entpair_distant[entpair]['word'].size(0)

            # Update support set representation
            support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)

            # Log embeddings after Phase 1
            self.plot_embeddings(support_pos_rep, support_pos['label'], title=f"After Phase 1 (Iteration {snowball_iter + 1})")

            # Phase 2: Use classifier to expand support set further
            self._phase2_add_num = 0
            candidate = distant.get_random_candidate(self.pos_class, candidate_num_class, candidate_num_ins_per_class)

            # Infer candidate probabilities
            candidate_prob = self._infer(candidate, batch_size=self.args.infer_batch_size)
            pick_or_not = self.siamese_model.forward_infer_sort(
                support_pos, candidate, batch_size=self.args.infer_batch_size
            )

            # Add high-confidence instances to support set
            self._phase2_total = candidate['word'].size(0)
            for i in range(min(len(candidate_prob), sort_num2)):
                iid = pick_or_not[i][1]
                if (pick_or_not[i][0] > sort_threshold2) and (candidate_prob[iid] > sort_ori_threshold) and not (
                    candidate['id'][iid] in exist_id
                ):
                    exist_id[candidate['id'][iid]] = 1
                    self._phase2_add_num += 1
                    self._add_ins_to_vdata(support_pos, candidate, iid, label=1)

            # Update support set representation
            support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)

            # Final fine-tuning
            self._train_finetune_init()
            self._train_finetune(support_pos_rep)

            # Logging and evaluation
            if self.args.eval:
                self._forward_eval_binary(query, threshold)
                self._metric.append(np.array([self._f1, self._prec, self._recall]))
                print(f"Phase 2 added {self._phase2_add_num} instances out of {self._phase2_total}")

        # Final evaluation
        self._forward_eval_binary(query, threshold)

        return support_pos_rep


    def _forward_eval_binary(self, query, threshold=0.5):
        '''
        snowball process (eval)
        query: query set (raw data)
        threshold: ins with prob > threshold will be classified as positive
        return (accuracy at threshold, precision at threshold, recall at threshold, f1 at threshold, auc), 
        '''
        query_prob = self._infer(query, batch_size=self.args.infer_batch_size).cpu().detach().numpy()
        label = query['label'].cpu().detach().numpy()
        accuracy = float(np.logical_or(np.logical_and(query_prob > threshold, label == 1), np.logical_and(query_prob < threshold, label == 0)).sum()) / float(query_prob.shape[0])
        if (query_prob > threshold).sum() == 0:
            precision = 0
        else:
            precision = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((query_prob > threshold).sum())
        recall = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((label == 1).sum())
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = float(2.0 * precision * recall) / float(precision + recall)
        auc = sklearn.metrics.roc_auc_score(label, query_prob)
        if self.args.print_debug:
            # Calculate accuracy
            accuracy = float(np.logical_or(np.logical_and(query_prob > threshold, label == 1), np.logical_and(query_prob < threshold, label == 0)).sum()) / float(query_prob.shape[0])
    
            # Log accuracy
            self.accuracy_history.append(accuracy) 
            print('')
            sys.stdout.write('[EVAL] acc: {0:2.2f}%, prec: {1:2.2f}%, rec: {2:2.2f}%, f1: {3:1.3f}, auc: {4:1.3f}'.format(\
                    accuracy * 100, precision * 100, recall * 100, f1, auc) + '\r')
            sys.stdout.flush()
        self._accuracy = accuracy
        self._prec = precision
        self._recall = recall
        self._f1 = f1
        return (accuracy, precision, recall, f1, auc)

    def forward(self, support_pos, query, distant, pos_class, threshold=0.5, threshold_for_snowball=0.5):
        '''
        snowball process (train + eval)
        support_pos: support set (positive, raw data)
        support_neg: support set (negative, raw data)
        query: query set (raw data)
        distant: distant data loader
        pos_class: positive relation (name)
        threshold: ins with prob > threshold will be classified as positive
        threshold_for_snowball: distant ins with prob > th_for_snowball will be added to extended support set
        '''
        self.pos_class = pos_class 
        self.plot_metrics()  # Call to plot metrics after training
        self._forward_train(support_pos, query, distant, threshold=threshold)

    def init_10shot(self, Ws, bs):
        self.Ws = torch.stack(Ws, 0).transpose(0, 1) # (230, 16)
        self.bs = torch.stack(bs, 0).transpose(0, 1) # (1, 16)

    def eval_10shot(self, query):
        x = self.sentence_encoder(query)
        x = torch.matmul(x, self.Ws) + self.new_bias # (batch_size, 16)
        x = torch.sigmoid(x)
        _, pred = x.max(-1) # (batch_size)
        return self.__accuracy__(pred, query['label'])

