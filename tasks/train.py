import argparse
import time

from torch.optim import Optimizer
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from common.base import BaseModel
from common.config import *
from models.model import *
from util.data_util import *


class BaseTrainTask(BaseModel):
    def __init__(self, model_name: str, params: dict):
        super().__init__(f"Model: {model_name} training task", input_=True)
        self.model: nn.Module = None
        self.optimizer: Optimizer = None
        parser = argparse.ArgumentParser()
        for param, val in params.items():
            parser.add_argument(f"--{param}", action="append", default=val)
        self.params = parser.parse_args()
        np.random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        self.params.device = (
            torch.device('cuda:' + str(self.params.cuda) if int(self.params.cuda) >= 0 else 'cpu'))
        LOG_TRAIN.info(f"Using device: {self.params.device}")
        torch.cuda.set_device(self.params.cuda)
        (entity2id, relation2id,
         train_data, val_data, test_data) = load_data(self.params.data_dir, self.params.dataset)
        LOG_TRAIN.info("Dataset: {}, Training data {:04d}".format(self.params.dataset, len(train_data[0])))
        # TODO: different models
        if self.params.model in ['MF']:
            self.corpus = ConvECorpus(self.params, train_data, val_data, test_data, entity2id, relation2id)
        else:
            self.corpus = ConvKBCorpus(self.params, train_data, val_data, test_data, entity2id, relation2id)
        if self.params.text_features:
            text_features = load_text_data(TEXT_EMBEDDING_SAVE_DIR, self.params.dataset, "entity_text_embedding.pkl")
            self.params.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)
            text_r_features = load_text_data(TEXT_EMBEDDING_SAVE_DIR, self.params.dataset, "relation_text_embedding.pkl")
            self.params.relation_desp = F.normalize(torch.Tensor(text_r_features), p=2, dim=1)
        self.params.entity2id = entity2id
        self.params.relation2id = relation2id

    def init_model(self, *args, **kwargs) -> nn.Module:
        """Override this function to initialize model."""
        raise NotImplementedError

    def init_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        """Override this function to initialize optimizer."""
        raise NotImplementedError

    def exec_train_task(self):
        """Override this function to execute training task."""
        raise NotImplementedError

    def exec_input(self, *args, **kwargs):
        self.model = self.init_model(*args, **kwargs)
        self.optimizer = self.init_optimizer(*args, **kwargs)
        LOG_TRAIN.info("Model info:\n" + str(self.model))
        tot_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        LOG_TRAIN.info(f'Total number of parameters: {tot_params}')
        if self.params.cuda is not None and int(self.params.cuda) >= 0:
            self.model = self.model.to(self.params.device)

    def exec_process(self, *args, **kwargs):
        self.exec_train_task()

    def exec_output(self) -> nn.Module:
        if self.params.save:
            path = f"{self.params.save_dir}{self.params.dataset}/{self.params.model}_{self.params.epochs}.pth"
            torch.save(self.model.state_dict(), path)
            LOG_TRAIN.info(f"Model saved to {path}")
        return self.model

    @staticmethod
    def format_metrics(metrics, split):
        return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

    @staticmethod
    def has_improved(m1, m2):
        return (m1["Mean Rank"] > m2["Mean Rank"]) or (m1["Mean Reciprocal Rank"] < m2["Mean Reciprocal Rank"])

    @staticmethod
    def init_metric_dict():
        return {"Hits@100": -1, "Hits@10": -1, "Hits@3": -1, "Hits@1": -1,
                "Mean Rank": 100000, "Mean Reciprocal Rank": -1}


class GATTrainTask(BaseTrainTask):
    def __init__(self):
        super().__init__("GAT", TRAIN_CONFIG)

    def init_model(self, *args, **kwargs) -> nn.Module:
        return GAT(self.params)

    def init_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay
        )

    def exec_train_task(self):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=500,
            gamma=float(self.params.gamma)
        )
        self.corpus.batch_size = len(self.corpus.train_triples)
        self.corpus.neg_num = 2
        t = time.time()
        for epoch in range(self.params.epochs):
            self.model.train()
            np.random.shuffle(self.corpus.train_triples)
            for i in range(1):
                train_indices, train_values = self.corpus.get_batch(i)
                train_indices = torch.LongTensor(train_indices)
                if self.params.cuda is not None and int(self.params.cuda) >= 0:
                    train_indices = train_indices.to(self.params.device)
                self.optimizer.zero_grad()
                entity_embed, relation_embed = self.model.forward(self.corpus.train_adj_matrix, train_indices)
                loss = self.model.loss_func(train_indices, entity_embed, relation_embed)
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
            if epoch % 100 == 0:
                LOG_TRAIN.info("Epoch {} , cost time {}s".format(epoch, time.time() - t))
                t = time.time()


class TextEmbeddingTask(BaseModel):
    def __init__(self):
        super().__init__("Text Embedding Task", input_=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertModel.from_pretrained(TEXT_EMBEDDING_MODEL).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(TEXT_EMBEDDING_TOKENIZER)
        self.max_length = TEXT_EMBEDDING_MAX_LENGTH
        self.data_dir = TEXT_EMBEDDING_DATA_DIR
        self.save_dir = TEXT_EMBEDDING_SAVE_DIR
        self.dataset = TEXT_EMBEDDING_DATASET
        self.entity_embedding = []
        self.relation_embedding = []

    def generate_text_embedding(self, text_file):
        result = []
        text = []
        with open(f"{self.data_dir}{self.dataset}/{text_file}", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    text.append(line)
        for line in tqdm(text):
            tokens = self.tokenizer.encode(line, add_special_tokens=True,
                                           max_length=self.max_length, truncation=True,
                                           return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(tokens)
                pooled_output = outputs[1]
            result.append(pooled_output.cpu().numpy())
        return result

    def exec_process(self, *args, **kwargs):
        self.entity_embedding = []
        LOG_TRAIN.info("Generating entity text embedding...")
        self.entity_embedding = self.generate_text_embedding("entity_text.txt")
        LOG_TRAIN.info("Generating relation text embedding...")
        self.relation_embedding = self.generate_text_embedding("relation_text.txt")

    def exec_output(self):
        with open(f"{self.save_dir}{self.dataset}/entity_text_embedding.pkl", "wb") as f:
            pickle.dump(self.entity_embedding, f)
        with open(f"{self.save_dir}{self.dataset}/relation_text_embedding.pkl", "wb") as f:
            pickle.dump(self.relation_embedding, f)


class MFTrainTask(BaseTrainTask):
    def __init__(self):
        super().__init__("Multimodal Fusion", TRAIN_CONFIG)

    def init_model(self, *args, **kwargs) -> nn.Module:
        return MultimodalFusionModule(self.params)

    def init_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay
        )

    def exec_train_task(self):
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.params.gamma)
        if self.params.encoder:
            model_gat = GAT(self.params)
            model_gat = model_gat.to(self.params.device)
            model_gat.load_state_dict(
                torch.load(f'./data/models/{self.params.dataset}/GAT_3000.pth'), strict=False)
            pickle.dump(model_gat.final_entity_embeddings.detach().cpu().numpy(),
                        open('./data/models/' + self.params.dataset + '/gat_entity_vec.pkl', 'wb'))
            pickle.dump(model_gat.final_relation_embeddings.detach().cpu().numpy(),
                        open('./data/models/' + self.params.dataset + '/gat_relation_vec.pkl', 'wb'))
        self.model = self.model.to(self.params.device)

        best_val_metrics = self.init_metric_dict()
        best_test_metrics = self.init_metric_dict()
        self.corpus.batch_size = self.params.batch_size
        self.corpus.neg_num = self.params.neg_num

        for epoch in range(self.params.epochs):
            self.model.train()
            epoch_loss = []
            t = time.time()
            self.corpus.shuffle()
            for batch_num in range(self.corpus.max_batch_num):
                self.optimizer.zero_grad()
                train_indices, train_values = self.corpus.get_batch(batch_num)
                train_indices = torch.LongTensor(train_indices)
                if self.params.cuda is not None and int(self.params.cuda) >= 0:
                    train_indices = train_indices.to(self.params.device)
                    train_values = train_values.to(self.params.device)
                output = self.model.forward(train_indices)
                loss = self.model.loss_func(output, train_values)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.data.item())
            lr_scheduler.step()

            if (epoch + 1) % self.params.eval_freq == 0:
                LOG_TRAIN.info("Epoch {:04d} , average loss {:.4f} , epoch_time {:.4f}".format(
                    epoch + 1, sum(epoch_loss) / len(epoch_loss), time.time() - t))
                self.model.eval()
                with torch.no_grad():
                    val_metrics = self.corpus.get_validation_pred(self.model, 'test')
                if val_metrics['Mean Reciprocal Rank'] > best_test_metrics['Mean Reciprocal Rank']:
                    best_test_metrics['Mean Reciprocal Rank'] = val_metrics['Mean Reciprocal Rank']
                if val_metrics['Mean Rank'] < best_test_metrics['Mean Rank']:
                    best_test_metrics['Mean Rank'] = val_metrics['Mean Rank']
                if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
                    best_test_metrics['Hits@1'] = val_metrics['Hits@1']
                if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
                    best_test_metrics['Hits@3'] = val_metrics['Hits@3']
                if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
                    best_test_metrics['Hits@10'] = val_metrics['Hits@10']
                if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
                    best_test_metrics['Hits@100'] = val_metrics['Hits@100']
                LOG_TRAIN.info(' '.join(['Epoch: {:04d}'.format(epoch + 1),
                                         self.format_metrics(val_metrics, 'test')]) + "\n")
        LOG_TRAIN.info('Optimization Finished!')
        if not best_test_metrics:
            self.model.eval()
            with torch.no_grad():
                best_test_metrics = self.corpus.get_validation_pred(self.model, 'test')
        LOG_TRAIN.info(' '.join(['Val set results:',
                                 self.format_metrics(best_val_metrics, 'val')]))
        LOG_TRAIN.info(' '.join(['Test set results:',
                                 self.format_metrics(best_test_metrics, 'test')]))
