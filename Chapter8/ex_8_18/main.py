import torch
# 静态模型的保存和载入
from torchvision.models import resnet18
m = resnet18(pretrained=True)
# 将模型从动态图转换为静态图
static_model = torch.jit.trace(m, torch.randn(1, 3, 224, 224))
# 保存模型
torch.jit.save(static_model, "resnet18.pt")
# 读取模型
static_model = torch.load("resnet18.pt")

# 导出到ONNX
from torchvision.models import resnet18
# 需要使用pip install onnx安装onnx的Python接口
import onnx
m = resnet18(pretrained=True)
torch.onnx.export(m, torch.randn(1, 3, 224, 224), 
                  "resnet18.onnx", verbose=True)
# 用onnx读入模型
m = onnx.load("resnet18.onnx")
# 检查模型正确性
onnx.checker.check_model(m)
# 打印计算图
onnx.helper.printable_graph(m.graph)
