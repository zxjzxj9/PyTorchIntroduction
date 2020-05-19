""" 该代码仅为演示函数签名所用，并不能实际运行
"""

save_info = { # 保存的信息
    "iter_num": iter_num,  # 迭代步数 
    "optimizer": optimizer.state_dict(), # 优化器的状态字典
    "model": model.state_dict(), # 模型的状态字典
}
# 保存信息
torch.save(save_info, save_path)
# 载入信息
save_info = torch.load(save_path)
optimizer.load_state_dict(save_info["optimizer"])
model.load_state_dict(sae_info["model"])