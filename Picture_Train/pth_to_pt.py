"""
该程序使用的是resnet34网络，将 .pth 权重文件转换为 .pt TorchScript 格式
"""
from resnet import resnet34
import torch
import torch.nn as nn

# 创建 ResNet34 模型
model = resnet34()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 修改这里为你的类别数

# 加载权重（修改为你的权重文件名）
checkpoint = torch.load('resnet34-1Net.pth',
                        map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        weights_only=True)
model.load_state_dict(checkpoint, strict=False)

model.eval()
# 将模型转换为TorchScript
example_input = torch.rand(1, 3, 224, 224)  # ResNet标准输入尺寸
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("resnet34-1Net.pt")
print('Finished Model Conversion')
