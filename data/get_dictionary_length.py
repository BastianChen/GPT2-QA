import config as cfg

# dictionary_path = cfg.DICTIONARY_TEST_PATH
dictionary_path = "dictionary/dictionary.txt"
with open(dictionary_path, "r+", encoding="utf-8") as file:
    strs = file.read().strip().split()
    print(len(strs))
