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
                        # for i in range(300):
                        #     if i == 0:
                        #         self.dataset.append(words[words_length - cfg.pos_num - 1:])
                        #         self.type_num.append(type_words[words_length - cfg.pos_num - 1:])
                        #     else:
                        #         self.dataset.append(words[words_length - cfg.pos_num - 1 - i:-i])
                        #         self.type_num.append(type_words[words_length - cfg.pos_num - 1 - i:-i])
                    # print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text_index = torch.tensor(self.dataset[item])
        type_index = torch.tensor(self.type_num[item])
        # print(text_index.shape)
        # print(type_index.shape)
        data = torch.stack([text_index, type_index], -1)
        return data[0:-1], data[1:]
        # return text_index[0:-1], text_index[1:], type_index[0:-1], type_index[1:]


if __name__ == '__main__':
    myDataset = MyDataset()
    # print(len(myDataset))
    # print(myDataset[0][0])
    # print(myDataset[1][0].shape)
    data, label = myDataset[202]
    # print(data)
    # print(label)
    print(data.shape)
    print(label.shape)
