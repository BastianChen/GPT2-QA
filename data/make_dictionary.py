import os
import config as cfg

'''获取所有的字'''

# text_path = cfg.TRAIN_TEXT_PATH
# dictionary_path = cfg.DICTIONARY_TRAIN_PATH
# text_path = cfg.TEST_TEXT_PATH
# dictionary_path = cfg.DICTIONARY_TEST_PATH
text_path = "original_text"
dictionary_path = "dictionary/dictionary.txt"

words = set()

for filename in os.listdir(text_path):
    with open(os.path.join(text_path, filename), "r+", encoding="utf-8") as file:
        word = file.read(1)
        while word:
            if word == '\n' or word == '\r' or word == ' ':
                # words.add('[SEP]')
                pass
            else:
                words.add(word)
            word = file.read(1)

with open(dictionary_path, "w+", encoding="utf-8") as f:
    f.write("[SEQ] [UNK] [PAD] [END] ♫ ♬ ")
    f.write(" ".join(words))
    f.flush()
