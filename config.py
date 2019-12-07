block_num = 6  # 定义解码器的数量
head_num = 12  # 定义几个多头注意力
embed_dim = 768  # 输入数据的长度
vocab_num = 2607  # 字典字数
pos_num = 500  # 输出内容最大长度
type_num = 4  # 句编码，可以用4个句子
stride = 5

# 训练过拟合版本时使用的路径
ORIGINAL_PATH = r"data/tokenized/original"
TYPE_PATH = r"data/tokenized/type_num"
DICTIONARY_PATH = "data/dictionary/dictionary.txt"

# 训练集JSON文件路径
TRAIN_JSON_PATH = r"D:\sample\baike_qa2019\baike_qa_train.json"
# 训练集文档保存路径
TRAIN_TEXT_PATH = r"original_text/train"
# 测试集JSON文件路径
TEST_JSON_PATH = r"D:\sample\baike_qa2019\baike_qa_valid.json"
# 测试集文档保存路径
TEST_TEXT_PATH = r"original_text/test"

# 训练集保存字典路径
DICTIONARY_TRAIN_PATH = r"dictionary/train/dictionary.txt"
# 测试集保存字典路径
DICTIONARY_TEST_PATH = r"dictionary/test/dictionary.txt"

# 编译训练集文档保存路径
TOKENIZED_TRAIN_ORIGINAL = r"data/tokenized/train/original"
TOKENIZED_TRAIN_TYPE = r"data/tokenized/train/type_num"
TOKENNIZE_TRAIN_ORIGINAL = r"tokenized/train/original"
TOKENIZE_TRAIN_TYPE = r"tokenized/train/type_num"

# 编译测试集文档保存路径
TOKENIZED_TEST_ORIGINAL = r"data/tokenized/test/original"
TOKENIZED_TEST_TYPE = r"data/tokenized/test/type_num"
TOKENNIZE_TEST_ORIGINAL = r"tokenized/test/original"
TOKENIZE_TEST_TYPE = r"tokenized/test/type_num"
