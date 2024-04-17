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
                pickle.load(open(args.save_dir + args.dataset + '/entity2vec.pkl', 'rb'))).float())
            self.relation_embeddings = nn.Parameter(torch.from_numpy(
                pickle.load(open(args.save_dir + args.dataset + '/relation2vec.pkl', 'rb'))).float())
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
        if TRAIN_CONFIG["cuda"] is not None and TRAIN_CONFIG["cuda"] >= 0:
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


class MultimodalFusionModule(nn.Module):
    def __init__(self, args):
        super(MultimodalFusionModule, self).__init__()
        self.device = args.device
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open(args.save_dir + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(),
                freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(
                    pickle.load(open(args.save_dir + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(
                    pickle.load(open(args.save_dir + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0),
                freeze=False)

        # relation_desp_tensor = torch.tensor(args.relation_desp).to(self.device)
        # self.txt_relation_embeddings = nn.Embedding.from_pretrained(relation_desp_tensor, freeze=False)
        # self.txt_relation_embeddings.weight.requires_grad = False

        txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
        txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
        txt = txt.view(txt.size(0), -1)
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)

        self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        self.dim = args.dim
        self.TuckER_S = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_D = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(args.dim, args.r_dim)
        self.Mutan_MM_E = MutanLayer(args.dim, 2)
        self.Mutan_MM_R = MutanLayer(args.dim, 2)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bce_loss = nn.BCELoss()

    @staticmethod
    def contrastive_loss(s_embed, t_embed):
        s_embed, t_embed = (s_embed / torch.norm(s_embed), t_embed / torch.norm(t_embed))
        pos_st = torch.sum(s_embed * t_embed, dim=1, keepdim=True)
        neg_s = torch.matmul(s_embed, s_embed.t())
        neg_t = torch.matmul(t_embed, t_embed.t())
        neg_s = neg_s - torch.diag_embed(torch.diag(neg_s))
        neg_t = neg_t - torch.diag_embed(torch.diag(neg_t))
        pos = torch.mean(torch.cat([pos_st], dim=1), dim=1)
        neg = torch.mean(torch.cat([neg_s, neg_t], dim=1), dim=1)
        loss = torch.mean(F.softplus(neg - pos))
        return loss

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        e_txt_embed = self.txt_entity_embeddings(head)
        r_txt_embed = self.txt_relation_embeddings(relation)

        e_mm_embed = self.Mutan_MM_E(e_embed, e_txt_embed)
        r_mm_embed = self.Mutan_MM_R(r_embed, r_txt_embed)

        pred_s = self.TuckER_S(e_embed, r_embed)
        pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)
        pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)

        pred_s = torch.mm(pred_s, self.entity_embeddings.weight.transpose(1, 0))
        pred_d = torch.mm(pred_d, self.txt_entity_embeddings.weight.transpose(1, 0))
        pred_mm = torch.mm(pred_mm, self.Mutan_MM_E(self.entity_embeddings.weight,
                                                    self.txt_entity_embeddings.weight).transpose(1, 0))

        pred_s = torch.sigmoid(pred_s)
        pred_d = torch.sigmoid(pred_d)
        pred_mm = torch.sigmoid(pred_mm)
        return [pred_s, pred_d, pred_mm]

    def loss_func(self, output, target):
        loss_s = self.bce_loss(output[0], target)
        loss_d = self.bce_loss(output[1], target)
        loss_mm = self.bce_loss(output[2], target)
        return loss_s + loss_d + loss_mm
