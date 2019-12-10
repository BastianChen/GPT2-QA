import os
import json
import torch
import config as cfg


# 解析json文件并保存
def resolveJson(path, save_path):
    file = open(path, "r", encoding="UTF-8")
    contents = file.readlines()
    for content in contents:
        # words = ["♪"]
        words = ['♫']
        fileJson = json.loads(content)
        # category = fileJson["category"].strip()
        title = fileJson["title"].strip()
        desc = fileJson["desc"].strip()
        answer = fileJson["answer"].strip()
        # word = category + '♫' + title + '♩' + desc + '♬' + answer + 'Ψ'
        word = title + '♩' + desc + '♬' + answer + 'Ψ'
        words.append(word)
        with open(save_path, "a+", encoding="utf-8") as file_save:
            file_save.write("".join(words))


if __name__ == '__main__':
    # resolveJson(cfg.TEST_JSON_PATH, cfg.TEST_TEXT_PATH)
    resolveJson(cfg.TRAIN_JSON_PATH, cfg.TRAIN_TEXT_PATH)
    # file = open(cfg.TRAIN_TEXT_PATH, "r", encoding="UTF-8")
    # print(file.read(500))
    pass
