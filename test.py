import torch

# a = torch.tensor([[[5, 1], [79, 1], [139, 1], [244, 1], [90, 1], [65, 1], [6, 2], [328, 2], [380, 2], [208, 2],
#                    [298, 2], [63, 2], [165, 2], [97, 2], [364, 2], [228, 2], [216, 2], [352, 2], [246, 2], [87, 2],
#                    [126, 2], [59, 2], [164, 2], [306, 2], [177, 2], [256, 2], [21, 2], [199, 2], [389, 2], [15, 2],
#                    [306, 2], [15, 2], [364, 2], [228, 2], [164, 2], [396, 2], [164, 2], [280, 2], [376, 2], [3, 2],
#                    [7, 3], [164, 3], [306, 3], [177, 3], [256, 3], [21, 3], [199, 3], [389, 3], [15, 3], [306, 3],
#                    [15, 3], [364, 3], [228, 3], [164, 3], [396, 3], [164, 3], [280, 3], [376, 3], [109, 3], [109, 3],
#                    [287, 3], [287, 3], [210, 3], [342, 3], [341, 3], [216, 3], [396, 3], [163, 3], [129, 3], [177, 3],
#                    [282, 3], [135, 3], [164, 3], [143, 3], [326, 3], [177, 3], [376, 3], [342, 3], [169, 3], [216, 3],
#                    [33, 3], [8, 4]]])
#
# print(a.shape)

# output = torch.tensor([[7]])
# output = torch.stack([output, torch.tensor([[2]])], -1)
# print(output)

# age = input("请输入您的年龄？")
# print(type(age))

# vocab = torch.tensor(
#             [[[0, 0], [6, 2], [393, 2],
#               [351, 2], [160, 2], [159, 2], [83, 2], [74, 2], [87, 2], [126, 2], [388, 2], [327, 2], [345, 2],
#               [3, 2], [7, 3], [393, 3], [351, 3], [160, 3], [159, 3], [83, 3], [74, 3], [87, 3], [126, 3],
#               [388, 3], [327, 3], [345, 3], [8, 4]]])
# print(vocab.shape)
# position = torch.tensor([range(82)])
# print(position.shape)

for i in range(0, 100, 5):
    print(i)
