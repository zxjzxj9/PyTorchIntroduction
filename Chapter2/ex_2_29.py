""" 本代码金威示例代码，演示如何使用PyTorch的学习率衰减功能
"""

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
