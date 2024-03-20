import pickle
from argparse import Namespace

from models.layer import *


class GAT(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        self.device = args.device
        self.num_nodes = len(args.entity2id)
        self.entity_in_dim = args.dim
        self.entity_out_dim = args.dim
        self.num_relation = len(args.relation2id)
        self.relation_in_dim = args.dim
        self.relation_out_dim = args.dim
        self.n_heads_GAT = args.n_heads
        self.neg_num = args.neg_num

        self.drop_GAT = args.dropout
        self.alpha = args.alpha

        # Initial Embedding
        self.entity_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.entity_in_dim))
        self.relation_embeddings = nn.Parameter(torch.randn(self.num_relation, self.relation_in_dim))
        if args.pre_trained:
            self.entity_embeddings = nn.Parameter(torch.from_numpy(
                pickle.load(open('datasets/' + args.dataset + '/entity2vec.pkl', 'rb'))).float())
            self.relation_embeddings = nn.Parameter(torch.from_numpy(
                pickle.load(open('datasets/' + args.dataset + '/relation2vec.pkl', 'rb'))).float())
        # Final output Embedding
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim * self.n_heads_GAT))
        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.relation_out_dim * self.n_heads_GAT))

        self.sp_gat = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim,
                            self.drop_GAT, self.alpha, self.n_heads_GAT)

        self.W_entities = nn.Parameter(torch.zeros(size=(self.entity_in_dim, self.entity_out_dim * self.n_heads_GAT)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, adj, train_indices):
        edge_list = adj[0]
        if CUDA:
            edge_list = edge_list.to(self.device)

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        self.relation_embeddings.data = F.normalize(
            self.relation_embeddings.data, p=2, dim=1).detach()

        mask_indices = torch.unique(train_indices[:, 2]).to(self.device)
        mask = torch.zeros(self.entity_embeddings.shape[0]).to(self.device)
        mask[mask_indices] = 1.0

        out_entity, out_relation = self.sp_gat(self.entity_embeddings, self.relation_embeddings, edge_list)
        out_entity = F.normalize(self.entity_embeddings.mm(self.W_entities)
                                 + mask.unsqueeze(-1).expand_as(out_entity) * out_entity, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity.data
        self.final_relation_embeddings.data = out_relation.data

        return out_entity, out_relation

    def loss_func(self, train_indices, entity_embeddings, relation_embeddings):
        len_pos_triples = int(train_indices.shape[0] / (int(self.neg_num) + 1))
        pos_triples = train_indices[:len_pos_triples]
        neg_triples = train_indices[len_pos_triples:]
        pos_triples = pos_triples.repeat(int(self.neg_num), 1)

        source_embeds = entity_embeddings[pos_triples[:, 0]]
        relation_embeds = relation_embeddings[pos_triples[:, 1]]
        tail_embeds = entity_embeddings[pos_triples[:, 2]]
        x = source_embeds + relation_embeds - tail_embeds
        pos_norm = torch.norm(x, p=1, dim=1)

        source_embeds = entity_embeddings[neg_triples[:, 0]]
        relation_embeds = relation_embeddings[neg_triples[:, 1]]
        tail_embeds = entity_embeddings[neg_triples[:, 2]]
        x = source_embeds + relation_embeds - tail_embeds
        neg_norm = torch.norm(x, p=1, dim=1)

        y = -torch.ones(int(self.neg_num) * len_pos_triples).to(self.device)
        loss = F.margin_ranking_loss(pos_norm, neg_norm, y, margin=1.0)
        return loss
