""" 本代码仅供半精度模型训练的饿参考
"""

from apex.fp16_utils import *
from apex import amp, optimizers

model = Model()
model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
model, optimizer = amp.initialize(model, optimizer,
                               opt_level=args.opt_level,
                               keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                               loss_scale=args.loss_scale)
# ...
loss = criterion(output, target)
optimizer.zero_grad()

with amp.scale_loss(loss, optimizer) as scaled_loss:
     scaled_loss.backward()
optimizer.step()