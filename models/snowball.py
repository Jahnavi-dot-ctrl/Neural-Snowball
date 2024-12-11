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
class Siamese(nn.Module):
    def __init__(self, sentence_encoder, hidden_size=230, drop_rate=0.5, pre_rep=None, euc=False, c=1.0):
        super(Siamese, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.hidden_size = hidden_size
        
        # Modified linear layer to handle different input shapes
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, 1)
        )
        
        self.cost = nn.BCEWithLogitsLoss()
        self.euc = euc
        self.c = c  # Curvature parameter for hyperbolic space

        # Tracking variables for metrics
        self.loss_history = []
        self.accuracy_history = []
        self.similarity_scores = []

    def _hyperbolic_distance(self, u, v):
        """Compute hyperbolic distance between two points."""
        eps = 1e-5
        u_norm = torch.clamp(torch.norm(u, dim=-1), max=1 - eps)
        v_norm = torch.clamp(torch.norm(v, dim=-1), max=1 - eps)
        diff = u - v
        diff_norm = torch.norm(diff, dim=-1)
        return 2 * torch.arctanh(diff_norm) / np.sqrt(self.c)

    def forward(self, data, num_size, num_class, threshold=0.5):
        # Encode input
        x = self.sentence_encoder(data).contiguous().view(num_class, num_size, -1)
        
        # Split data
        x1 = x[:, :num_size // 2].contiguous().view(-1, self.hidden_size)
        x2 = x[:, num_size // 2:].contiguous().view(-1, self.hidden_size)
        y1 = x[:num_class // 2, :].contiguous().view(-1, self.hidden_size)
        y2 = x[num_class // 2:, :].contiguous().view(-1, self.hidden_size)

        # Prepare labels
        label = torch.zeros((x1.size(0) + y1.size(0))).long().cuda()
        label[:x1.size(0)] = 1
        
        # Concatenate data
        z1 = torch.cat([x1, y1], 0)
        z2 = torch.cat([x2, y2], 0)

        # Compute distance
        if self.euc:
            # Euclidean distance
            dis = torch.abs(z1 - z2)
        else:
            # Hyperbolic distance
            dis = self._hyperbolic_distance(z1, z2).unsqueeze(1)

        # Combine features and pass through network
        combined_features = torch.cat([z1, z2], dim=1)
        score = self.fc(combined_features).squeeze()

        # Compute loss and metrics
        self._loss = self.cost(score, label.float())
        pred = torch.sigmoid(score) > threshold
        
        # Compute accuracy and precision/recall
        self._accuracy = torch.mean((pred == label).float())
        pred_np = pred.cpu().detach().numpy()
        label_np = label.cpu().detach().numpy()
        
        self._prec = float(np.logical_and(pred_np == 1, label_np == 1).sum()) / (pred_np.sum() + 1e-8)
        self._recall = float(np.logical_and(pred_np == 1, label_np == 1).sum()) / (label_np.sum() + 1e-8)

        # Track metrics
        self.loss_history.append(self._loss.item())
        self.accuracy_history.append(self._accuracy.item())
        self.similarity_scores.append(self._prec)

        return self._loss

    def forward_infer_sort(self, support, query, batch_size=32, threshold=0.5):
        """
        Inference method for sorting/comparing support and query instances
        
        Args:
            support: Support set instances
            query: Query set instances
            batch_size: Batch size for processing
            threshold: Similarity threshold
        
        Returns:
            Tensor indicating which query instances are picked
        """
        # Encode support and query instances
        support_encoded = self.sentence_encoder(support)
        query_encoded = self.sentence_encoder(query)
        
        # Compute similarities
        similarities = []
        for i in range(0, query_encoded.size(0), batch_size):
            batch_query = query_encoded[i:i+batch_size]
            batch_sims = []
            
            for j in range(0, support_encoded.size(0), batch_size):
                batch_support = support_encoded[j:j+batch_size]
                
                # Compute pairwise similarities
                combined_features = torch.cat([
                    batch_query.unsqueeze(1).repeat(1, batch_support.size(0), 1).view(-1, self.hidden_size),
                    batch_support.unsqueeze(0).repeat(batch_query.size(0), 1, 1).view(-1, self.hidden_size)
                ], dim=1)
                
                scores = torch.sigmoid(self.fc(combined_features)).view(batch_query.size(0), batch_support.size(0))
                batch_sims.append(scores)
            
            batch_sims = torch.cat(batch_sims, dim=1)
            similarities.append(batch_sims)
        
        similarities = torch.cat(similarities, dim=0)
        
        # Pick instances above threshold
        pick_or_not = (similarities.max(dim=1).values > threshold).float()
        
        return pick_or_not

    def encode(self, dataset, batch_size=0):
        self.sentence_encoder.eval()
        with torch.no_grad():
            if batch_size == 0:
                return self.sentence_encoder(dataset)
            
            total_length = dataset['word'].size(0)
            max_iter = (total_length + batch_size - 1) // batch_size
            
            encoded_batches = []
            for it in range(max_iter):
                start = batch_size * it
                end = min(batch_size * (it + 1), total_length)
                scope = list(range(start, end))
                
                batch_data = {
                    'word': dataset['word'][scope],
                    'mask': dataset['mask'][scope]
                }
                
                if 'pos1' in dataset:
                    batch_data['pos1'] = dataset['pos1'][scope]
                    batch_data['pos2'] = dataset['pos2'][scope]
                
                encoded_batches.append(self.sentence_encoder(batch_data).detach())
            
            return torch.cat(encoded_batches, 0)

    def plot_metrics(self):
        """Plot loss, accuracy, and similarity scores."""
        epochs = range(1, len(self.loss_history) + 1)

        plt.figure(figsize=(15, 5))

        # Loss subplot
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.loss_history, label='Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Accuracy subplot
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.accuracy_history, label='Accuracy', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        # Similarity Scores subplot
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.similarity_scores, label='Similarity', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Similarity Scores')
        plt.title('Similarity Scores over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

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

    # def __loss__(self, logits, label):
    #     onehot_label = torch.zeros(logits.size()).cuda()
    #     onehot_label.scatter_(1, label.view(-1, 1), 1)
    #     return self.cost(logits, onehot_label)

    # def __loss__(self, logits, label):
    #     return self.cost(logits, label)

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

    def _train_finetune(self, data_repre, learning_rate=None, weight_decay=1e-5):
        '''
        train finetune classifier with given data
        data_repre: sentence representation (encoder's output)
        label: label
        '''
        
        self.train()

        optimizer = self.optimizer
        if learning_rate is not None:
            optimizer = optim.Adam([self.new_W, self.new_bias], learning_rate, weight_decay=weight_decay)

        # hyperparameters
        max_epoch = self.args.finetune_epoch
        batch_size = self.args.finetune_batch_size
        
        # dropout
        data_repre = self.drop(data_repre) 
        
        # train
        if self.args.print_debug:
            print('')
        for epoch in range(max_epoch):
            max_iter = data_repre.size(0) // batch_size
            if data_repre.size(0) % batch_size != 0:
                max_iter += 1
            order = list(range(data_repre.size(0)))
            random.shuffle(order)
            for i in range(max_iter):            
                x = data_repre[order[i * batch_size : min((i + 1) * batch_size, data_repre.size(0))]]
                # batch_label = label[order[i * batch_size : min((i + 1) * batch_size, data_repre.size(0))]]
                
                # neg sampling
                # ---------------------
                batch_label = torch.ones((x.size(0))).long().cuda()
                neg_size = int(x.size(0) * 1)
                neg = self.neg_loader.next_batch(neg_size)
                neg = self.encode(neg, self.args.infer_batch_size)
                x = torch.cat([x, neg], 0)
                batch_label = torch.cat([batch_label, torch.zeros((neg_size)).long().cuda()], 0)
                # ---------------------

                x = torch.matmul(x, self.new_W) + self.new_bias # (batch_size, 1)
                x = torch.sigmoid(x)

                # iter_loss = self.__loss__(x, batch_label.float()).mean()
                weight = torch.ones(batch_label.size(0)).float().cuda()
                weight[batch_label == 0] = self.args.finetune_weight #1 / float(max_epoch)
                iter_loss = (self.__loss__(x, batch_label.float()) * weight).mean()

                optimizer.zero_grad()
                iter_loss.backward(retain_graph=True)
                optimizer.step()
                if self.args.print_debug:
                    sys.stdout.write('[snowball finetune] epoch {0:4} iter {1:4} | loss: {2:2.6f}'.format(epoch, i, iter_loss) + '\r')
                    sys.stdout.flush()
        self.eval()

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
        '''
        snowball process (train)
        support_pos: support set (positive, raw data)
        support_neg: support set (negative, raw data)
        query: query set
        distant: distant data loader
        threshold: ins with prob > threshold will be classified as positive
        threshold_for_phase1: distant ins with prob > th_for_phase1 will be added to extended support set at phase1
        threshold_for_phase2: distant ins with prob > th_for_phase2 will be added to extended support set at phase2
        '''

        # hyperparameters
        snowball_max_iter = self.args.snowball_max_iter
        sys.stdout.flush()
        candidate_num_class = 20
        candidate_num_ins_per_class = 100
        
        sort_num1 = self.args.phase1_add_num
        sort_num2 = self.args.phase2_add_num
        sort_threshold1 = self.args.phase1_siamese_th
        sort_threshold2 = self.args.phase2_siamese_th
        sort_ori_threshold = self.args.phase2_cl_th

        # get neg representations with sentence encoder
        # support_neg_rep = self.encode(support_neg, batch_size=self.args.infer_batch_size)
        
        # init
        self._train_finetune_init()
        # support_rep = self.encode(support, self.args.infer_batch_size)
        support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
        # self._train_finetune(support_rep, support['label'])
        self._train_finetune(support_pos_rep)

        self._metric = []

        # copy
        original_support_pos = copy.deepcopy(support_pos)

        # snowball
        exist_id = {}
        if self.args.print_debug:
            print('\n-------------------------------------------------------')
        for snowball_iter in range(snowball_max_iter):
            if self.args.print_debug:
                print('###### snowball iter ' + str(snowball_iter))
            # phase 1: expand positive support set from distant dataset (with same entity pairs)

            ## get all entpairs and their ins in positive support set
            old_support_pos_label = support_pos['label'] + 0
            entpair_support = {}
            entpair_distant = {}
            for i in range(len(support_pos['id'])): # only positive support
                entpair = support_pos['entpair'][i]
                exist_id[support_pos['id'][i]] = 1
                if entpair not in entpair_support:
                    if 'pos1' in support_pos:
                        entpair_support[entpair] = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
                    else:
                        entpair_support[entpair] = {'word': [], 'mask': [], 'id': []}
                self._add_ins_to_data(entpair_support[entpair], support_pos, i)
            
            ## pick all ins with the same entpairs in distant data and choose with siamese network
            self._phase1_add_num = 0 # total number of snowball instances
            self._phase1_total = 0
            for entpair in entpair_support:
                raw = distant.get_same_entpair_ins(entpair) # ins with the same entpair
                if raw is None:
                    continue
                if 'pos1' in support_pos:
                    entpair_distant[entpair] = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': []}
                else:
                    entpair_distant[entpair] = {'word': [], 'mask': [], 'id': [], 'entpair': []}
                for i in range(raw['word'].size(0)):
                    if raw['id'][i] not in exist_id: # don't pick sentences already in the support set
                        self._add_ins_to_data(entpair_distant[entpair], raw, i)
                self._dataset_stack_and_cuda(entpair_support[entpair])
                self._dataset_stack_and_cuda(entpair_distant[entpair])
                if len(entpair_support[entpair]['word']) == 0 or len(entpair_distant[entpair]['word']) == 0:
                    continue

                
                pick_or_not = self.siamese_model.forward_infer_sort(entpair_support[entpair], entpair_distant[entpair], batch_size=self.args.infer_batch_size)
                
                # pick_or_not = self.siamese_model.forward_infer_sort(original_support_pos, entpair_distant[entpair], threshold=threshold_for_phase1)
                # pick_or_not = self._infer(entpair_distant[entpair]) > threshold
      
                # -- method B: use sort --
                for i in range(min(len(pick_or_not), sort_num1)):
                    if pick_or_not[i][0] > sort_threshold1:
                        iid = pick_or_not[i][1]
                        self._add_ins_to_vdata(support_pos, entpair_distant[entpair], iid, label=1)
                        exist_id[entpair_distant[entpair]['id'][iid]] = 1
                        self._phase1_add_num += 1
                self._phase1_total += entpair_distant[entpair]['word'].size(0)
            '''
            if 'pos1' in support_pos:
                candidate = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': []}
            else:
                candidate = {'word': [], 'mask': [], 'id': [], 'entpair': []}

            self._phase1_add_num = 0 # total number of snowball instances
            self._phase1_total = 0
            for entpair in entpair_support:
                raw = distant.get_same_entpair_ins(entpair) # ins with the same entpair
                if raw is None:
                    continue
                for i in range(raw['word'].size(0)):
                    if raw['id'][i] not in exist_id: # don't pick sentences already in the support set
                        self._add_ins_to_data(candidate, raw, i)

            if len(candidate['word']) > 0:
                self._dataset_stack_and_cuda(candidate)
                pick_or_not = self.siamese_model.forward_infer_sort(support_pos, candidate, batch_size=self.args.infer_batch_size)
                    
                for i in range(min(len(pick_or_not), sort_num1)):
                    if pick_or_not[i][0] > sort_threshold1:
                        iid = pick_or_not[i][1]
                        self._add_ins_to_vdata(support_pos, candidate, iid, label=1)
                        exist_id[candidate['id'][iid]] = 1
                        self._phase1_add_num += 1
                self._phase1_total += candidate['word'].size(0)
            '''
            ## build new support set
            
            # print('---')
            # for i in range(len(support_pos['entpair'])):
            #     print(support_pos['entpair'][i])
            # print('---')
            # print('---')
            # for i in range(support_pos['id'].size(0)):
            #     print(support_pos['id'][i])
            # print('---')

            support_pos_rep = self.encode(support_pos, batch_size=self.args.infer_batch_size)
            # support_rep = torch.cat([support_pos_rep, support_neg_rep], 0)
            # support_label = torch.cat([support_pos['label'], support_neg['label']], 0)
            
            ## finetune
            # print("Fine-tune Init")
            self._train_finetune_init()
            self._train_finetune(support_pos_rep)
            if self.args.eval:
                self._forward_eval_binary(query, threshold)
            # self._metric.append(np.array([self._f1, self._prec, self._recall]))
            if self.args.print_debug:
                print('\nphase1 add {} ins / {}'.format(self._phase1_add_num, self._phase1_total))

            # phase 2: use the new classifier to pick more extended support ins
            self._phase2_add_num = 0
            candidate = distant.get_random_candidate(self.pos_class, candidate_num_class, candidate_num_ins_per_class)

            ## -- method 1: directly use the classifier --
            candidate_prob = self._infer(candidate, batch_size=self.args.infer_batch_size)
            ## -- method 2: use siamese network --

            pick_or_not = self.siamese_model.forward_infer_sort(support_pos, candidate, batch_size=self.args.infer_batch_size)

            ## -- method A: use threshold --
            '''
            self._phase2_total = candidate_prob.size(0)
            for i in range(candidate_prob.size(0)):
                # if (candidate_prob[i] > threshold_for_phase2) and not (candidate['id'][i] in exist_id):
                if (pick_or_not[i]) and (candidate_prob[i] > threshold_for_phase2) and not (candidate['id'][i] in exist_id):
                    exist_id[candidate['id'][i]] = 1 
                    self._phase2_add_num += 1
                    self._add_ins_to_vdata(support_pos, candidate, i, label=1)
            '''

            ## -- method B: use sort --
            self._phase2_total = candidate['word'].size(0)
            for i in range(min(len(candidate_prob), sort_num2)):
                iid = pick_or_not[i][1]
                if (pick_or_not[i][0] > sort_threshold2) and (candidate_prob[iid] > sort_ori_threshold) and not (candidate['id'][iid] in exist_id):
                    exist_id[candidate['id'][iid]] = 1 
                    self._phase2_add_num += 1
                    self._add_ins_to_vdata(support_pos, candidate, iid, label=1)

            ## build new support set
            support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
            # support_rep = torch.cat([support_pos_rep, support_neg_rep], 0)
            # support_label = torch.cat([support_pos['label'], support_neg['label']], 0)

            ## finetune
            # print("Fine-tune Init")
            self._train_finetune_init()
            self._train_finetune(support_pos_rep)
            if self.args.eval:
                self._forward_eval_binary(query, threshold)
                self._metric.append(np.array([self._f1, self._prec, self._recall]))
                if self.args.print_debug:
                    print('\nphase2 add {} ins / {}'.format(self._phase2_add_num, self._phase2_total))

        self._forward_eval_binary(query, threshold)
        if self.args.print_debug:
            print('\nphase2 add {} ins / {}'.format(self._phase2_add_num, self._phase2_total))

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
