import os
import config as cfg

'''根据字典索引进行编码'''
# 训练集路径
# tokenized_path = cfg.TOKENNIZE_TRAIN_ORIGINAL
# type_num_path = cfg.TOKENIZE_TRAIN_TYPE
# text_path = cfg.TRAIN_TEXT_PATH
# dictionary_path = cfg.DICTIONARY_TRAIN_PATH
# 测试集路径
# tokenized_path = cfg.TOKENNIZE_TEST_ORIGINAL
# type_num_path = cfg.TOKENIZE_TEST_TYPE
# text_path = cfg.TEST_TEXT_PATH
# dictionary_path = cfg.DICTIONARY_TEST_PATH
tokenized_path = cfg.ORIGINAL_PATH
type_num_path = cfg.TYPE_PATH
text_path = "original_text"
dictionary_path = "dictionary/dictionary.txt"

if not os.path.exists(tokenized_path):
    os.makedirs(tokenized_path)

with open(dictionary_path, "r+", encoding="utf-8") as file:
    dics = file.read().strip().split()

count = 0
for filename in os.listdir(text_path):
    # original_file = open(os.path.join(tokenized_path, "{}.txt".format(count)), "a", encoding="utf-8")
    # type_file = open(os.path.join(type_num_path, "{}.txt".format(count)), "a", encoding="utf-8")
    # type_num为1表示是问题类型，2是标题，3是描述，4是回答 ,0是开始符号
    # type_num = "0"
    type_num = ""
    f_path = os.path.join(text_path, filename)
    with open(f_path, "r+", encoding="utf-8") as f:
        # indexs = ["0"]
        # type_indexs = ["0"]
        indexs = [""]
        type_indexs = [""]
        word = f.read(1)
        while word:
            if word == '\n' or word == '\r' or word == '\t' or ord(word) == 12288:
                # indexs.append("1")
                indexs.append("0")
            elif word == ' ':
                # indexs.append("3")
                indexs.append("2")
            # elif word == '♪':
            #     indexs.append("5")
            #     type_num = "1"
            elif word == '♫':
                # indexs.append("5")
                # type_num = "1"
                indexs.append("4")
                type_num = "0"
            # elif word == '♩':
            #     # indexs.append("6")
            #     # type_num = "2"
            #     indexs.append("5")
            #     type_num = "1"
            elif word == '♬':
                # indexs.append("7")
                # type_num = "3"
                indexs.append("5")
                type_num = "1"
            elif word == 'Ψ':
                # indexs.append("4")
                indexs.append("3")
            else:
                try:
                    indexs.append(str(dics.index(word)))
                except:
                    # indexs.append("2")
                    indexs.append("1")
            type_indexs.append(type_num)
            word = f.read(1)
        count += 1

    with open(os.path.join(tokenized_path, "{}.txt".format(count)), "w+", encoding="utf-8") as df:
        df.write(" ".join(indexs))
    with open(os.path.join(type_num_path, "{}.txt".format(count)), "w+", encoding="utf-8") as df:
        df.write(" ".join(type_indexs))
