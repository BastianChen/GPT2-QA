from nets import GPT2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datasets import MyDataset
import config as cfg
from tensorboardX import SummaryWriter


def weight_init(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Trainer:
    def __init__(self, save_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = GPT2().to(self.device)
        # self.weight_file_bak = os.path.join("weights", "apt2_k_bak.pt")
        # self.weight_file = os.path.join("weights", "apt2_k.pt")
        self.train_data = DataLoader(MyDataset(), batch_size=2, shuffle=True)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss = nn.CrossEntropyLoss()
        self.save_path = save_path
        if os.path.exists(self.save_path):
            self.net.load_state_dict(torch.load(self.save_path))
        else:
            self.net.apply(weight_init)
        self.net.train()

    def train(self):
        epoch = 1
        loss_new = 1000
        writer = SummaryWriter()
        while True:
            for i, (data, labels) in enumerate(self.train_data):
                data, labels = data.to(self.device), labels.to(self.device)
                # position = torch.arange(0, data.shape[1])[None, :].repeat(data.shape[0], 1).to(self.device)
                position = torch.arange(0, data.shape[1]).repeat(data.shape[0], 1).to(self.device)
                # type = torch.arange(0, data.shape[1])[None, :].repeat(data.shape[0], 1).to(self.device)

                # print(data.shape)
                # print(position.shape)
                # print(position)

                output = self.net(data, position).reshape(-1, cfg.vocab_num)
                labels = labels[:, :, 0].reshape(-1)
                loss = self.loss(output, labels)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                writer.add_scalar("loss", loss, epoch)

                if i % 10 == 0:
                    print("epoch:{0},i:{1},loss:{2}".format(epoch, i, loss.item()))

                if loss.item() < loss_new:
                    loss_new = loss.item()
                    torch.save(self.net.state_dict(), self.save_path)
            epoch += 1


if __name__ == '__main__':
    trainer = Trainer("models/net_qa.pth")
    trainer.train()
