"""
# 非极大值抑制算法实现
"""

import os 
from model.config import cfg 
from nms.gpu_nms import gpu_nms 
from nms.cpu_nms import cpu_nms 

def nms(dets, thresh, force_cpu=False):
  """Dispatch to either CPU or GPU NMS implementations."""

  if dets.shape[0] == 0:
    return []
  if cfg.USE_GPU_NMS and not force_cpu:
    return gpu_nms(dets, thresh, device_id=0)
  else:
    return cpu_nms(dets, thresh)
