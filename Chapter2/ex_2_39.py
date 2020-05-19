""" 该代码仅为演示函数签名和所用方法，并不能实际运行
"""

torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
model = … # 定义模型
model = model.cuda()
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]) # 数据并行
output = model(input_var)
