import os
from torch.utils.data import Dataset
import config as cfg
import torch


class MyDataset(Dataset):
    def __init__(self):
        self.dataset = []
        self.type_num = []
        for filename in os.listdir(cfg.ORIGINAL_PATH):
            with open(os.path.join(cfg.ORIGINAL_PATH, filename), "r+") as file:
                words = [int(word) for word in file.read().split()]
                words_length = len(words)
                type_file = open(os.path.join(cfg.TYPE_PATH, filename), "r+")
                type_words = [int(type_num) for type_num in type_file.read().split()]
                start = 0
                while words_length - start > cfg.pos_num + 1:
                    self.dataset.append(words[start:start + cfg.pos_num + 1])
                    self.type_num.append(type_words[start:start + cfg.pos_num + 1])
                    start += cfg.stride
                else:
                    if words_length > cfg.pos_num + 1:
                        self.dataset.append(words[words_length - cfg.pos_num - 1:])
                        self.type_num.append(type_words[words_length - cfg.pos_num - 1:])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text_index = torch.tensor(self.dataset[item])
        type_index = torch.tensor(self.type_num[item])
        data = torch.stack([text_index, type_index], -1)
        return data[0:-1], data[1:]


if __name__ == '__main__':
    myDataset = MyDataset()
    # print(len(myDataset))
    # print(myDataset[0][0])
    # print(myDataset[1][0].shape)
    data, label = myDataset[0]
    # print(data)
    # print(label)
    print(data.shape)
    print(label.shape)
