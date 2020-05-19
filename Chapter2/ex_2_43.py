""" 该代码仅为演示函数签名所用，并不能实际运行
"""

if not args.multiprocessing_distributed or (args.multiprocessing_distributed
    and args.rank % ngpus_per_node == 0):
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)
