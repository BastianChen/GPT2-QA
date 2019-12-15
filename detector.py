from nets import GPT2
import torch
import config as cfg
import numpy as np


class Detector:
    def __init__(self, net_path, dictionary_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.dictionary_path = dictionary_path
        self.net = GPT2().to(self.device)
        self.net.load_state_dict(torch.load(net_path))
        self.net.eval()

    def getVocab(self, title, answer):
        with open(self.dictionary_path, "r+", encoding="utf-8") as file:
            dics = file.read().strip().split()
            words = title + answer
            vocab_list = []
            title_len = len(title)
            title_type = torch.zeros(title_len, dtype=torch.long)
            answer_type = torch.ones(1, dtype=torch.long)
            type_num = torch.cat((title_type, answer_type), -1)
            type_num = type_num.reshape(type_num.size(0), 1)
            for word in words:
                if word == '\n' or word == '\r' or word == '\t' or ord(word) == 12288:
                    vocab_list.append(0)
                elif word == ' ':
                    vocab_list.append(2)
                elif word == '♫':
                    vocab_list.append(4)
                elif word == '♬':
                    vocab_list.append(5)
                elif word == 'Ψ':
                    vocab_list.append(3)
                else:
                    try:
                        vocab_list.append(dics.index(word))
                    except:
                        vocab_list.append(2)
            vocab = torch.tensor(np.stack(vocab_list), dtype=torch.long)
            vocab = vocab.reshape(vocab.size(0), 1)
            self.vocab = torch.cat((vocab, type_num), -1).unsqueeze(0).to(self.device)
            self.position = torch.tensor([range(self.vocab.shape[1])]).to(self.device)

    def detect(self):
        title = input("请输入问题：")
        # desc = input("请输入问题的详细描述：")
        # title = '脚上长了一些泡，不痛不痒比较硬。'
        # desc = '昨天好像吃到什么不太的东西了,一直拉肚子,搞的我睡觉也睡不好,索性就起来了, 我该吃些什么东西调理下'
        title = '♫' + title
        answer = '♬'
        self.getVocab(title.strip(), answer.strip())
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
            output = torch.stack([output, torch.tensor([[1]]).to(self.device)], -1)
            output_list.append(output[..., 0])
            self.vocab = torch.cat([self.vocab, output], dim=-2).to(self.device)
            self.position = torch.tensor([range(self.vocab.size(1))]).to(self.device)
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


if __name__ == '__main__':
    detector = Detector("models/net_qa_new.pth", cfg.DICTIONARY_PATH)
    detector.detect()
