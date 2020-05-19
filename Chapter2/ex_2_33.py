""" 该代码仅为演示类的构造方法所用，并不能实际运行
"""

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
            assert end > start, \
"this example code only works with end >= start"
            self.start = start
            self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # 单进程数据载入
            iter_start = self.start
            iter_end = self.end
        else:  # 多进程，分割数据
            per_worker = int(math.ceil((self.end - self.start) \
                            / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))
