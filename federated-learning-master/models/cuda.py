import torch

# 创建一个张量
input_tensor = torch.randn(1, 3, 224, 224)

# 调用函数并传入张量
is_acceptable = torch.cudnn_is_acceptable(input_tensor)
print(is_acceptable)