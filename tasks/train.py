import argparse
import time

from torch.optim import Optimizer

from common.base import BaseModel
from common.config import *
from models.model import *
from util.data_util import *


class BaseTrainTask(BaseModel):
    def __init__(self, model_name: str, params: dict):
        super().__init__(f"Model: {model_name} training task", input_=True)
        self.model: nn.Module | None = None
        self.optimizer: Optimizer | None = None
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
         img_features, text_features,
         train_data, val_data, test_data) = load_data(self.params.data_dir, self.params.dataset)
        LOG_TRAIN.info("Dataset: {}, Training data {:04d}".format(self.params.dataset, len(train_data[0])))
        # TODO: different models
        self.corpus = ConvKBCorpus(self.params, train_data, val_data, test_data, entity2id, relation2id)

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
