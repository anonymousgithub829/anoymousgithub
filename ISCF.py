import pyximport
pyximport.install()
import math, os, pickle, random, sys, argparse
from time import time
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
import copy
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModel, Trainer
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from function import setup_seed, convert_csr_to_sparse_tensor_inputs, create_adj_mat, csr_to_user_dict, pad_sequences, pad_sequences_3d, dropout_sparsetensor, SparseDropout
from evaluate import *

# 定义一个层复制函数，将每一层的结构执行深拷贝，并返回list形式
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 先对输入值x进行reshape一下，然后交换在维度1,2进行交换
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Model(nn.Module):
    def __init__(self, params, training_user, training_item, summary):
        super(Model, self).__init__()
        self.params = params
        # self.pt_model = AutoModel.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = params.device
        self.n_users = params.n_users
        self.n_items = params.n_items
        self.n_critics = params.n_critics

        # parameters of LightGCN
        self.keep_prob = params.keep_prob
        self.ssl_ratio = params.ssl_ratio
        self.embedding_size = params.embedding_size
        self.n_layers = params.n_layers
        self.training_user, self.training_item = training_user, training_item
        self.weights = dict()
        self.emb_user_critic = nn.Embedding(self.n_users + self.n_critics, self.embedding_size)
        self.emb_item = nn.Embedding(self.n_items, self.embedding_size)
        self.sub_mat = {}
        self.norm_adj = create_adj_mat(self.training_user, self.training_item, params.n_users + params.n_critics, params.n_items, params.ssl_ratio, is_subgraph=False)
        temp_indices, temp_values, temp_shape = convert_csr_to_sparse_tensor_inputs(self.norm_adj, self.device)
        self.adj_mat = torch.sparse.FloatTensor(temp_indices, temp_values, temp_shape)

        # parameters of VAE
        self.anneal = 0.0
        self.dims = [self.n_items, 600, self.embedding_size]
        self.input_dropout = nn.Dropout(1-self.params.keep_prob)
        self.decoder = nn.Embedding(self.n_items, self.dims[-1])
        # self.sparsedropout = SparseDropout(self.params.keep_prob)
        self.vae_weights = nn.ParameterDict()
        for k in range(1, len(self.dims)):
            if k != len(self.dims) - 1:
                self.vae_weights['W_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(self.dims[k-1], self.dims[k])).to(self.device)
                self.vae_weights['b_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(self.dims[k])).to(self.device)
            else:
                self.vae_weights['W_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(self.dims[k-1], 2*self.dims[k])).to(self.device)
                self.vae_weights['b_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(2*self.dims[k])).to(self.device)
            nn.init.xavier_normal_(self.vae_weights['W_encoder_%d' % k])
            nn.init.trunc_normal_(self.vae_weights['b_encoder_%d' % k], std=0.001, a=-0.002, b=0.002)

        # parameters of Text
        # self.attn_u = MultiHeadedAttention(5, self.embedding_size, dropout=self.params.drop)
        # self.attn_v = MultiHeadedAttention(5, self.embedding_size, dropout=self.params.drop)
        self.summary = summary
        self.summary_fc = nn.Linear(self.summary.shape[1], self.embedding_size, bias=True)
        self.emb_text_user = nn.Embedding(self.n_users, self.embedding_size)
        self.fc1 = nn.Linear(2 * self.embedding_size, 2 * self.embedding_size)
        self.fc2 = nn.Linear(2 * self.embedding_size, self.params.hidden_size)
        self.emb_text_critic = nn.Embedding(self.n_critics, self.embedding_size)
        self.fc1_critic = nn.Linear(2 * self.embedding_size, 2 * self.embedding_size)
        self.fc2_critic = nn.Linear(2 * self.embedding_size, self.params.hidden_size)
        self.fc_att1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_att2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_att3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.dropout = nn.Dropout(p=self.params.drop)
        self.fc_coef1 = nn.Linear(self.embedding_size, 2 * self.embedding_size)
        self.fc_coef2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_coef3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.summary_fc2 = nn.Linear(self.summary.shape[1], self.dims[-1], bias=False)

    def forward(self, users, items, input_ph, review_users, review_items, batch_review, item_review_critics, mask, critic_ids, item_ids, critic_scores, batch_critic_review, critics, critic_input_ph, is_train=1):
        # LightGCN model
        self.weights['user_embedding'] = self.emb_user_critic(torch.arange(params.n_users + params.n_critics, dtype=torch.long, device=self.device))
        self.weights['item_embedding'] = self.emb_item(torch.arange(params.n_items, dtype=torch.long, device=self.device))

        self.ua_embeddings, self.ia_embeddings = self.create_lightgcn_SSL_embed()
        batch_u_embeddings, batch_i_embeddings = torch.index_select(self.ua_embeddings, 0, users), self.ia_embeddings
        batch_c_embeddings = torch.index_select(self.weights['user_embedding'], 0, critics)
        
        # Multi-VAE model
        self.vae_i_emb = self.summary_fc2(self.summary) + self.decoder(torch.arange(params.n_items, dtype=torch.long, device=self.device))
        self.vae_u_emb = self.create_VAE_embed(input_ph)

        # Text model
        self.summary_emb = self.summary_fc(self.summary)
        text_v_emb = torch.index_select(self.summary_emb, 0, review_items)
        text_u_emb = self.emb_text_user(review_users)
        nlp_embdeddings = torch.cat([text_v_emb, text_u_emb], -1)
        nlp_embdeddings = self.fc1(nlp_embdeddings)
        nlp_embdeddings = self.fc2(nlp_embdeddings)
        
        mask = mask.unsqueeze(1)
        critic_c_emb = self.emb_text_critic(item_review_critics)
        critic_c_emb = torch.cat((self.summary_emb.unsqueeze(1), critic_c_emb), dim=1)
        critic_v_emb, _ = attention((self.fc_att1(self.summary_emb).unsqueeze(1)), self.fc_att2(critic_c_emb), self.fc_att3(critic_c_emb), mask=mask.unsqueeze(-2).expand(mask.size(0), mask.size(1), mask.size(1)), dropout=self.dropout)
        critic_v_emb = critic_v_emb[:, 0, :]
        text_c_emb_critic = self.emb_text_critic(critic_ids)
        text_v_emb_critic = torch.index_select(self.summary_emb, 0, item_ids)
        pred_critic = torch.mean(text_c_emb_critic * text_v_emb_critic, dim=1)
        nlp_embdeddings_critic = torch.cat([text_c_emb_critic, text_v_emb_critic], -1)
        nlp_embdeddings_critic = self.fc1_critic(nlp_embdeddings_critic)
        nlp_embdeddings_critic = self.fc2_critic(nlp_embdeddings_critic)

        # combine embeddings
        joint_u_emb = self.vae_u_emb + self.emb_text_user(users)
        joint_v_emb = self.vae_i_emb + critic_v_emb
        logits = torch.mm(joint_u_emb, joint_v_emb.t())
        
        # # 计算对比学习Loss
        normalize_user_emb1 = F.normalize(batch_u_embeddings, p=2, dim=1)
        normalize_user_emb2 = F.normalize(joint_u_emb, p=2, dim=1)
        normalize_all_user_emb2 = normalize_user_emb2
        pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2), 1)
        ttl_score_user = torch.mm(normalize_user_emb1, normalize_all_user_emb2.t())
        pos_score_user = torch.exp(pos_score_user / self.params.ssl_temp)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.params.ssl_temp), 1)
        ssl_loss_user = - torch.mean(torch.log(pos_score_user / ttl_score_user))

        normalize_item_emb1 = F.normalize(torch.index_select(batch_i_embeddings, 0, items), p=2, dim=1)
        normalize_item_emb2 = F.normalize(torch.index_select(joint_v_emb, 0, items), p=2, dim=1)
        normalize_all_item_emb2 = normalize_item_emb2
        pos_score_item = torch.sum(torch.mul(normalize_item_emb1, normalize_item_emb2), 1)
        ttl_score_item = torch.mm(normalize_item_emb1, normalize_all_item_emb2.t())
        pos_score_item = torch.exp(pos_score_item / self.params.ssl_temp)
        ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.params.ssl_temp), 1)
        ssl_loss_item = - torch.mean(torch.log(pos_score_item / ttl_score_item))

        self.ssl_loss = self.params.ssl_reg * (ssl_loss_user + ssl_loss_item)

        # 计算隐向量正则化Loss
        self.emb_loss = 0
        for temp_w in (batch_u_embeddings, batch_i_embeddings, batch_c_embeddings):
            self.emb_loss += torch.mean(torch.norm(temp_w, p=2, dim=1))
        for temp_w in (text_u_emb, text_v_emb, text_c_emb_critic):
            self.emb_loss += torch.mean(torch.norm(temp_w, p=2, dim=1))
        self.emb_loss += torch.mean(torch.norm(critic_v_emb, p=2, dim=1))
        self.emb_loss = self.params.reg * self.emb_loss


        # 计算预测Loss
        log_softmax_var = F.log_softmax(logits)
        logits_gcn = torch.mm(batch_u_embeddings, batch_i_embeddings.t())
        log_softmax_var_gcn = F.log_softmax(logits_gcn)
        logits_gcn_critic = torch.mm(batch_c_embeddings, batch_i_embeddings.t())
        log_softmax_var_gcn_critic = F.log_softmax(logits_gcn_critic)
        
        self.rating_loss = - torch.mean(torch.sum(log_softmax_var * input_ph, 1))
        self.gcn_loss = 0
        self.gcn_loss = - self.params.eta * torch.mean(torch.sum(log_softmax_var_gcn * input_ph, 1))
        self.gcn_loss += - self.params.eta * torch.mean(torch.sum(log_softmax_var_gcn_critic * critic_input_ph, 1))

        criterion = nn.MSELoss(reduction='none')
        self.nlp_loss = self.params.nlp_reg * torch.mean(torch.sum(criterion(nlp_embdeddings, batch_review), 1))
        self.nlp_loss += self.params.nlp_reg * torch.mean(torch.sum(criterion(nlp_embdeddings_critic, batch_critic_review), 1))
        criterion2 = nn.MSELoss(reduction='mean')
        self.critic_loss = self.params.cri_reg * criterion2(pred_critic, critic_scores)

        # print(self.rating_loss, self.KL, self.emb_loss, self.ssl_loss, self.nlp_loss, self.gcn_loss, self.critic_loss)

        self.loss = (self.rating_loss + self.anneal * self.KL) + self.emb_loss + self.ssl_loss + self.nlp_loss + self.gcn_loss + self.critic_loss

        return self.loss, self.rating_loss, self.emb_loss, self.ssl_loss, self.nlp_loss
    
    def get_prediction(self, input_ph=None, item_review_critics=None, mask=None):
        critic_c_emb = self.emb_text_critic(item_review_critics)
        critic_c_emb = torch.cat((self.summary_emb.unsqueeze(1), critic_c_emb), dim=1)
        critic_v_emb, _ = attention((self.fc_att1(self.summary_emb).unsqueeze(1)), self.fc_att2(critic_c_emb), self.fc_att3(critic_c_emb), mask=mask.unsqueeze(-2).expand(mask.size(0), mask.size(1), mask.size(1)), dropout=self.dropout)
        critic_v_emb = critic_v_emb[:, 0, :]
        critic_v_emb = torch.squeeze(critic_v_emb, dim=1)
        
        joint_u_emb = self.create_VAE_embed(input_ph) + self.emb_text_user(torch.arange(params.n_users, dtype=torch.long, device=self.device))
        joint_v_emb = self.vae_i_emb + critic_v_emb
        batch_ratings = torch.mm(joint_u_emb, joint_v_emb.t())

        return batch_ratings.cpu().numpy()
    
    def loss_corr(self, x, nnodes=None):
        if nnodes is None:
            nnodes = x.shape[0]
        idx = np.random.choice(x.shape[0], int(np.sqrt(nnodes)))
        x = x[idx]
        x = x - x.mean(0)
        cov = x.t() @ x
        I_k = torch.eye(x.shape[1]).to(self.device) / np.sqrt(x.shape[1])
        loss = torch.norm(cov / torch.norm(cov) - I_k)
        return loss

    def create_lightgcn_SSL_embed(self):
        ego_embeddings = torch.cat([self.weights['user_embedding'], self.weights['item_embedding']], 0)
        all_embeddings = [ego_embeddings]
        
        for k in range(1, self.n_layers + 1):
            ego_embeddings = torch.sparse.mm(self.adj_mat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings = torch.mean(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users + self.n_critics, self.n_items], 0)

        return u_g_embeddings.to(self.device), i_g_embeddings.to(self.device)
    
    def create_VAE_embed(self, input_ph):
        h = F.normalize(input_ph, p=2, dim=1)
        h = self.input_dropout(h)
        for k in range(1, len(self.dims)):
            h = torch.mm(h, self.vae_weights['W_encoder_%d' % k]) + self.vae_weights['b_encoder_%d' % k]
            if k != len(self.dims) - 1:
                h = F.tanh(h)
            else:
                mu_q = h[:, :self.dims[-1]]
                logvar_q = h[:, self.dims[-1]:]
                std_q = torch.exp(0.5 * logvar_q)
                self.KL = -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=1).mean()
        
        epsilon = torch.randn_like(std_q)
        users = mu_q + self.training * epsilon * std_q
        return users

parser = argparse.ArgumentParser()
parser.add_argument('-dev', action='store', dest='dev', default='2')
parser.add_argument('-print', action='store', dest='print', default=10, type=int)
parser.add_argument('-keep_prob', type=float, default=0.5)
parser.add_argument('-drop', type=float, default=0.2)
parser.add_argument('-reg', type=float, default=15.0)
parser.add_argument('-nlp_reg', type=float, default=10.0)
parser.add_argument('-cri_reg', type=float, default=10.0)
parser.add_argument('-corr_reg', type=float, default=0.0)
parser.add_argument('-ssl', type=float, default=0.1)
parser.add_argument('-ssl_temp', type=float, default=0.2)
parser.add_argument('-ssl_reg', type=float, default=7.0)
parser.add_argument('-batch_size', type=int, default=2048, help='input batch size')
parser.add_argument('-hidden_size', type=int, default=384, help='hidden state size of transformers module')
parser.add_argument('-embed_dim', type=int, default=150, help='the dimension of item embedding')
parser.add_argument('-epochs', type=int, default=3000, help='the number of epochs to train for')
parser.add_argument('-layers', type=int, default=3, help='the number of epochs to train for')
parser.add_argument('-lr', type=float, default=0.003, help='learning rate')
parser.add_argument('-log', type=str, default='')
parser.add_argument('-step', action='store', dest='step', default=20, type=int)
parser.add_argument('-eta', type=float, default=10)
parser.add_argument('-max_len', type=int, default=20)
args = parser.parse_args()


setup_seed(20)
class Params:
    def __init__(self):
        self.print_epoch = args.print
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embed_dim
        self.batch_size = args.batch_size
        self.keep_prob = args.keep_prob
        self.drop = args.drop
        self.reg = args.reg
        self.nlp_reg = args.nlp_reg
        self.corr_reg = args.corr_reg
        self.cri_reg = args.cri_reg
        self.device = torch.device('cuda:' + args.dev if torch.cuda.is_available() else 'cpu')
        self.n_layers = args.layers
        self.ssl_ratio = args.ssl
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.eta = args.eta
        self.epochs = args.epochs
        self.hidden_dims = [100, 200, 300]
        self.model_name = "down-paraphrase-multilingual-MiniLM-L12-v2"
        self.n_users, self.n_items, self.n_critics = 0, 0, 0
        self.max_len = args.max_len

# 处理user数据
params = Params()
data_handle_dir = '../data/‘
train_list = pd.read_csv(data_handle_dir + "train{}.csv".format(0), header=None, encoding='UTF-8')
valid_list = pd.read_csv(data_handle_dir + "probe{}.csv".format(0), header=None, encoding='UTF-8')
train_list = train_list[[0, 1, 4, 2]].values
valid_list = valid_list[[0, 1, 4, 2]].values
valid_rating = (valid_list[:, 3].astype(np.float32))/np.max(valid_list[:, 3]).astype(np.float32)
valid_list = valid_list[valid_rating >= 0.7, :]
params.n_users, params.n_items = np.max(np.concatenate((train_list, valid_list))[:, 0:2], axis=0) + 1
posprobe1 = defaultdict(list)
for line in valid_list:
    if line[2] == 0:
        posprobe[line[0]].append(line[1])

train_rating = (train_list[:, 3].astype(np.float32))/np.max(train_list[:, 3]).astype(np.float32)
train_matrix = sp.csr_matrix((train_rating, (train_list[:, 0], train_list[:, 1])), dtype='float32', shape=(params.n_users, params.n_items))
valid_matrix = sp.csr_matrix((np.ones_like(valid_list[:, 0]), (valid_list[:, 0], valid_list[:, 1])), dtype='float32', shape=(params.n_users, params.n_items))
user_pos_dict = csr_to_user_dict(train_matrix)
num_trainings = sum([len(item) for u, item in user_pos_dict.items()])
user_pos_dict = {u: np.array(item) for u, item in user_pos_dict.items()}
users_list, items_list = torch.tensor(np.arange(params.n_users), dtype=torch.long).to(params.device), torch.tensor(np.arange(params.n_items), dtype=torch.long).to(params.device)

summary = pd.read_csv(data_handle_dir + "pretraininfo{}.csv".format(0), header=None, encoding='UTF-8').values
summary = torch.FloatTensor(summary).to(params.device)
review = pd.read_csv(data_handle_dir + "pretrainuserreview{}.csv".format(0), header=None, encoding='UTF-8').values
review_dict = {(train_list[i, 0], train_list[i, 1]) : i for i in range(len(train_list))}

# 处理critic数据
critic_data = pd.read_csv(data_handle_dir + "cross_critic_review{}.csv".format(0), header=0, encoding='UTF-8')
critic_data = critic_data[critic_data['score'] != 'None']
critic_data['score'] = pd.to_numeric(critic_data['score'], downcast='integer')
critic_data = critic_data[['CRITIC_ID', 'ITEM_ID', 'domain', 'score']].values
n_critic_data = len(critic_data)
params.n_critics = np.max(critic_data[:, 0]) + 1
critic_rating = (critic_data[:, 3].astype(np.float32))/np.max(critic_data[:, 3]).astype(np.float32)
critic_matrix = sp.csr_matrix((critic_rating, (critic_data[:, 0], critic_data[:, 1])), dtype='float32', shape=(params.n_critics, params.n_items))
item_critic_dict = csr_to_user_dict(critic_matrix.transpose())
critic_review = pd.read_csv(data_handle_dir + "pretraincriticreview.csv", header=None, encoding='UTF-8').values
critic_review_dict = {(critic_data[i, 0], critic_data[i, 1]) : i for i in range(n_critic_data)}
criticids_list = torch.tensor(np.arange(params.n_critics), dtype=torch.long).to(params.device)

# 合并user和critic数据建立网络
critic_list = np.copy(critic_data)
critic_list[:, 0] = critic_list[:, 0] + params.n_users
user_critic_list = np.concatenate((train_list, critic_list), axis=0)
user_critic_rating = np.concatenate((train_rating, critic_rating), axis=0)
user_critic_matrix = sp.csr_matrix((user_critic_rating, (user_critic_list[:, 0], user_critic_list[:, 1])), dtype='float32', shape=(params.n_users + params.n_critics, params.n_items))
dok_matrix = user_critic_matrix.todok()
training_user, training_item = [], []
for (user, item), value in dok_matrix.items():
    training_user.append(user)
    training_item.append(item)

update_count = 0.0
model = Model(params, training_user, training_item, summary).to(params.device)
optimizer = optim.Adam(model.parameters(), args.lr)
# scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

input_ph_all = torch.FloatTensor(train_matrix.toarray()).to(params.device)
total_anneal_steps = args.step
negtrainmatrix = 99999 * train_matrix.toarray()
posprobe = csr_to_user_dict(valid_matrix)
recall, precision, map, ndcg = [], [], [], []
idxlist, idxitemlist, idxcriticlist, idxcriticidlist = list(range(params.n_users)), list(range(params.n_items)), list(range(n_critic_data)), list(range(params.n_critics))
n_batch = math.ceil(params.n_users / params.batch_size)
item_batch_size = math.ceil(params.n_items / n_batch)
critic_batch_size = math.ceil(n_critic_data / n_batch)
criticid_batch_size = math.ceil(params.n_critics / n_batch)
ones_tensor = torch.ones(params.n_items, 1, dtype=torch.long).to(params.device)
for epoch in range(1, params.epochs + 1):
    training_start_time = time()
    model.train()
    total_loss, total_rating_loss, total_ssl_loss, total_emb_loss, total_nlp_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    np.random.shuffle(idxlist)
    np.random.shuffle(idxitemlist)
    np.random.shuffle(idxcriticlist)
    np.random.shuffle(idxcriticidlist)

    for i in range(n_batch):
        st_idx = i * params.batch_size
        end_idx = min(st_idx + params.batch_size, params.n_users)
        st_item_idx = i * item_batch_size
        end_item_idx = min(st_item_idx + item_batch_size, params.n_items)
        users, items = users_list[idxlist[st_idx:end_idx]], items_list[idxitemlist[st_item_idx:end_item_idx]]
        input_ph = torch.FloatTensor(train_matrix[idxlist[st_idx:end_idx]].toarray()).to(params.device)
        users, items = torch.tensor(users, dtype=torch.long).to(params.device), torch.tensor(items, dtype=torch.long).to(params.device)

        review_items = np.concatenate([np.random.permutation(user_pos_dict[ii])[:params.max_len] for ii in idxlist[st_idx:end_idx]])
        review_users = np.concatenate([[ii] * min(len(user_pos_dict[ii]), params.max_len) for ii in idxlist[st_idx:end_idx]])
        batch_review = torch.FloatTensor(review[[review_dict[(review_users[ii], review_items[ii])] for ii in range(len(review_users))]]).to(params.device)
        review_users, review_items = torch.tensor(review_users, dtype=torch.long).to(params.device), torch.tensor(review_items, dtype=torch.long).to(params.device)

        item_review_critics = [np.random.permutation(item_critic_dict[ii])[:params.max_len] for ii in list(range(params.n_items))]
        item_review_critics = pad_sequences(item_review_critics, max_len=params.max_len)
        mask = torch.tensor((item_review_critics != 0).astype(np.int64), dtype=torch.long).to(params.device)
        mask = torch.cat((ones_tensor, mask), dim=1)
        item_review_critics = torch.tensor(item_review_critics, dtype=torch.long).to(params.device)
        st_critic_idx = i * critic_batch_size
        end_critic_idx = min(st_critic_idx + critic_batch_size, n_critic_data)
        batch_critic = critic_data[idxcriticlist[st_critic_idx:end_critic_idx]]
        batch_critic_review = torch.FloatTensor(critic_review[[critic_review_dict[(batch_critic[ii, 0], batch_critic[ii, 1])] for ii in range(len(batch_critic))]]).to(params.device)
        
        st_criticid_idx = i * criticid_batch_size
        end_criticid_idx = min(st_criticid_idx + criticid_batch_size, params.n_critics)
        critics = criticids_list[idxcriticidlist[st_criticid_idx:end_criticid_idx]] + params.n_users
        critic_input_ph = torch.FloatTensor(user_critic_matrix[idxcriticidlist[st_criticid_idx:end_criticid_idx]].toarray()).to(params.device)
        
        model.anneal = min(1.0, 1. * update_count / total_anneal_steps)
        update_count += 1
        optimizer.zero_grad()
        loss, rating_loss, emb_loss, ssl_loss, nlp_loss = model(users, items, input_ph, review_users, review_items, batch_review, item_review_critics, mask, torch.tensor(batch_critic[:, 0], 
            dtype=torch.long).to(params.device), torch.tensor(batch_critic[:, 1], dtype=torch.long).to(params.device), torch.tensor(critic_rating[idxcriticlist[st_critic_idx:end_critic_idx]], 
            dtype=torch.float32).to(params.device), batch_critic_review, critics, critic_input_ph, is_train=1)

        total_loss += loss
        total_rating_loss += rating_loss
        total_emb_loss += emb_loss
        total_ssl_loss += ssl_loss
        total_nlp_loss += nlp_loss

        loss.backward()
        optimizer.step()

    # scheduler.step(epoch=epoch)
    print('Epoch {} loss: {:.4f}, rating_loss: {:.4f}, emb_loss: {:.4f}, ssl_loss: {:.4f}, nlp_loss: {:.4f}, time: {} \n'.format(epoch, total_loss, total_rating_loss, total_emb_loss, total_ssl_loss, total_nlp_loss, time() - training_start_time))

    if epoch % params.print_epoch == 0:
        model.eval()
        test_start_time = time()
        with torch.no_grad():
            item_review_critics = [np.random.permutation(item_critic_dict[ii])[:params.max_len] for ii in list(range(params.n_items))]
            item_review_critics = pad_sequences(item_review_critics, max_len=params.max_len)
            mask = torch.tensor((item_review_critics != 0).astype(np.int64), dtype=torch.long).to(params.device)
            mask = torch.cat((ones_tensor, mask), dim=1)
            item_review_critics = torch.tensor(item_review_critics, dtype=torch.long).to(params.device)
            epcoh_rating = model.get_prediction(input_ph_all, item_review_critics, mask)
        epcoh_rating = epcoh_rating - negtrainmatrix
        test_start_time1 = time()
        recall_batch, precision_batch, map_batch, ndcg_batch = evaluate11(posprobe, epcoh_rating, [5, 10])
        print(precision_batch, recall_batch, map_batch, ndcg_batch, time() - test_start_time1)
        precision.append(precision_batch)
        recall.append(recall_batch)
        map.append(map_batch)
        ndcg.append(ndcg_batch)
        evaluation = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(map), pd.DataFrame(ndcg)], axis=1)
        filename = "dumper/cross" + args.log + "_" + str(args.keep_prob) + "_" + str(args.lr) + "_" + str(args.reg) + "_" + str(args.ssl_reg) + "_" + str(args.nlp_reg) + "_" + str(args.cri_reg) + "_" + str(args.step) + "_" + str(args.drop) + "_" + str(args.embed_dim)
        evaluation.to_csv(filename + ".csv", header=False, index=False)
        
    if epoch % 30 == 0:
        ckpt_dict = {'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
        torch.save(ckpt_dict, 'model/MCIR-epoch{}.pth.tar'.format(epoch))
    






















