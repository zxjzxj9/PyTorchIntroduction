""" 该代码仅为演示函数签名所用，并不能实际运行
"""

torch.nn.parallel.DistributedDataParallel(module, device_ids=None, 
    output_device=None, dim=0, broadcast_buffers=True, 
    process_group=None, bucket_cap_mb=25, 
    find_unused_parameters=False, check_reduction=False)