# DHCA-Net: Dual-Branch Hierarchical Weighted Contrast with Cross-Attention Network for Cervical Abnormal Cell Detection
Our approach uses mmdetection, some modules and code refer to mmdetection(https://github.com/open-mmlab/mmdetection)

## Datasets
The additional annotation dataset will be fully open-sourced after the publication of the paper.

## Method

Our overall framework is implemented in [mmdet/models/roi_heads/cascade_roi_head]. The implementation of the IPCA and HMWC modules are in [models/roi_heads/contractive_loss.py] and [models/roi_heads/feature_attention.py] respectively.


