import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch

print("当前可见的GPU数量:", torch.cuda.device_count())
print("当前使用的GPU名称:", torch.cuda.get_device_name(0))

# 随便建一个张量丢进显卡
a = torch.randn(1000, 1000).cuda()
print("成功将数据放入 GPU 4！学长没有限制我！")