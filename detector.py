from trainer import Trainer
from nets import GPT2
import torch
import random
import config as cfg
import numpy as np


class Detector:
    def __init__(self, net_path, dictionary_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.dictionary_path = dictionary_path
        self.net = GPT2().to(self.device)
        # self.vocab = torch.tensor([[[random.randint(0, 415), random.randint(0, 5)]]])
        # self.position = torch.tensor([[random.randint(0, 500)]])
        # self.vocab = torch.tensor(
        #     [[[5, 1], [203, 1], [324, 1], [254, 1], [146, 1], [244, 1], [82, 1], [60, 1], [146, 1], [247, 1],
        #       [244, 1], [403, 1], [120, 1], [146, 1], [6, 2], [164, 2], [269, 2], [67, 2], [368, 2], [56, 2],
        #       [214, 2], [126, 2], [77, 2], [53, 2], [26, 2], [270, 2], [196, 2], [102, 2], [102, 2], [102, 2],
        #       [3, 2], [8, 4]]])
        # self.position = torch.tensor([range(32)])
        # self.vocab = torch.tensor(
        #     [[[0, 0], [6, 2], [393, 2],
        #       [351, 2], [160, 2], [159, 2], [83, 2], [74, 2], [87, 2], [126, 2], [388, 2], [327, 2], [345, 2],
        #       [3, 2], [7, 3], [393, 3], [351, 3], [160, 3], [159, 3], [83, 3], [74, 3], [87, 3], [126, 3],
        #       [388, 3], [327, 3], [345, 3], [8, 4]]])
        # self.position = torch.tensor([range(27)])
        # self.vocab = torch.tensor(
        #     [[[5, 1], [79, 1], [139, 1], [244, 1], [90, 1], [65, 1], [6, 2], [328, 2], [380, 2], [208, 2],
        #       [298, 2], [63, 2], [165, 2], [97, 2], [364, 2], [228, 2], [216, 2], [352, 2], [246, 2], [87, 2],
        #       [126, 2], [59, 2], [164, 2], [306, 2], [177, 2], [256, 2], [21, 2], [199, 2], [389, 2], [15, 2],
        #       [306, 2], [15, 2], [364, 2], [228, 2], [164, 2], [396, 2], [164, 2], [280, 2], [376, 2], [3, 2],
        #       [7, 3], [164, 3], [306, 3], [177, 3], [256, 3], [21, 3], [199, 3], [389, 3], [15, 3], [306, 3],
        #       [15, 3], [364, 3], [228, 3], [164, 3], [396, 3], [164, 3], [280, 3], [376, 3], [109, 3], [109, 3],
        #       [287, 3], [287, 3], [210, 3], [342, 3], [341, 3], [216, 3], [396, 3], [163, 3], [129, 3], [177, 3],
        #       [282, 3], [135, 3], [164, 3], [143, 3], [326, 3], [177, 3], [376, 3], [342, 3], [169, 3], [216, 3],
        #       [33, 3], [8, 4]]])
        # self.position = torch.tensor([range(82)])
        self.net.load_state_dict(torch.load(net_path))
        self.net.eval()

    def getVocab(self, title, desc, answer):
        with open(self.dictionary_path, "r+", encoding="utf-8") as file:
            dics = file.read().strip().split()
            words = title + desc + answer
            vocab_list = []
            title_len = len(title)
            desc_len = len(desc)
            title_type = torch.ones(title_len, dtype=torch.long)
            desc_type = torch.ones(desc_len, dtype=torch.long)
            desc_type[:] = 2
            answer_type = torch.ones(1, dtype=torch.long)
            answer_type[:] = 3
            type_num = torch.cat((title_type, desc_type), -1)
            type_num = torch.cat((type_num, answer_type), -1)
            type_num = type_num.reshape(type_num.size(0), 1)
            for word in words:
                if word == '\n' or word == '\r' or word == '\t' or ord(word) == 12288:
                    vocab_list.append(1)
                elif word == ' ':
                    vocab_list.append(3)
                elif word == '♫':
                    vocab_list.append(5)
                elif word == '♩':
                    vocab_list.append(6)
                elif word == '♬':
                    vocab_list.append(7)
                elif word == 'Ψ':
                    vocab_list.append(4)
                else:
                    try:
                        vocab_list.append(dics.index(word))
                    except:
                        vocab_list.append(2)
            vocab = torch.tensor(np.stack(vocab_list), dtype=torch.long)
            vocab = vocab.reshape(vocab.size(0), 1)
            self.vocab = torch.cat((vocab, type_num), -1).unsqueeze(0).to(self.device)
            self.position = torch.tensor([range(self.vocab.size(0))]).to(self.device)

    def detect(self):
        # title = input("请输入问题：")
        # desc = input("请输入问题的详细描述：")
        title = '拉肚子以后,身体虚弱,应该吃些什么东西调养,有什么应该注意的?'
        desc = '昨天好像吃到什么不太的东西了,一直拉肚子,搞的我睡觉也睡不好,索性就起来了, 我该吃些什么东西调理下'
        title = '♫' + title
        desc = '♩' + desc
        answer = '♬'
        self.getVocab(title.strip(), desc.strip(), answer.strip())
        print("回答如下：")
        output_list = []
        for i in range(200):
            output = self.net(self.vocab, self.position)
            output = output[:, -1:]
            # 得到8个最大的值跟索引
            value, index = torch.topk(output, 8, dim=-1)
            # torch.multinomial()只能用于1维或2维的tensor
            value, index = value[0], index[0]
            value_index = torch.multinomial(torch.softmax(value, dim=-1), 1)
            output = index[0][value_index]
            # 效果与上一行代码相同
            # output = torch.gather(index, -1, v_index)
            # print(output.shape)
            output = torch.stack([output, torch.tensor([[3]]).to(self.device)], -1)
            output_list.append(output[..., 0])
            self.vocab = torch.cat([self.vocab, output], dim=-2).to(self.device)
            self.position = torch.tensor([range(self.vocab.size(1))]).to(self.device)
            # self.position = torch.tensor([range(i + 33)])
            # self.position = torch.tensor([range(i + 28)])
            # self.position = torch.tensor([range(i + 83)])
        with open(self.dictionary_path, "r+", encoding="utf-8") as dictionary:
            strs = dictionary.read().split()
            output_list = torch.stack(output_list)
            for index in output_list:
                if strs[index[0]] == "[SEQ]":
                    print()
                elif strs[index[0]] == "[PAD]":
                    print(" ", end="")
                elif strs[index[0]] == "[START]":
                    print()
                elif strs[index[0]] == "[END]":
                    print("end...")
                    break
                else:
                    print(strs[index[0]], end="")
            # for index in self.vocab[0]:
            #     if strs[index[0]] == "[SEQ]":
            #         print()
            #     elif strs[index[0]] == "[PAD]":
            #         print(" ", end="")
            #     elif strs[index[0]] == "[START]":
            #         print()
            #     elif strs[index[0]] == "[END]":
            #         print("end...")
            #         break
            #     else:
            #         print(strs[index[0]], end="")


if __name__ == '__main__':
    detector = Detector("models/net_qa.pth", cfg.DICTIONARY_PATH)
    detector.detect()
