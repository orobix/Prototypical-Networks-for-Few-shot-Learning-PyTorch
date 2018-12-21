from torch import optim

from prototypical_loss import PrototypicalLoss
from utils.dataloader import (load_meta_test_dataloader, load_meta_train_dataloaders)


class FewShotTrainingParameters:

    def __init__(self, model, sets):
        self.n_episodes = 100
        self.n_epochs = 10000
        self.l1_lambda = 0.1

        learning_rate = 1e-3
        n_support = 5
        n_query = 15
        samples_per_class = n_support + n_query
        n_ways = (5, 5)

        self.train_loader, self.valid_loader = load_meta_train_dataloaders(sets,
                                                                           samples_per_class=samples_per_class,
                                                                           n_episodes=self.n_episodes,
                                                                           classes_per_it=n_ways)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=learning_rate)

        self.criterion = PrototypicalLoss(n_support)

        # Reduce the learning rate by half every 2000 episodes
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=20,  # 20 epochs of 100 episodes
                                                   gamma=0.5)

        self.PATIENCE_LIMIT = 20


class FewShotTestParameters:

    def __init__(self, test_set):
        n_support = 5
        n_query = 15
        n_ways = 5
        samples_per_class = n_support + n_query

        self.criterion = PrototypicalLoss(n_support)
        self.n_episodes = 100
        self.n_epochs = 6

        self.test_loader = load_meta_test_dataloader(test_set,
                                                     samples_per_class=samples_per_class,
                                                     n_episodes=self.n_episodes,
                                                     classes_per_it=n_ways)
