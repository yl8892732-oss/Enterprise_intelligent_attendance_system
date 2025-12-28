import os
import tensorflow as tf
from modules.models import ArcFaceModel
from modules.utils import load_yaml

# 1. 加载你的配置文件（确保参数和训练时一致）
cfg = load_yaml('./configs/arc_res50.yaml')

# 2. 重新构建模型骨架 (推理模式 training=False)
model = ArcFaceModel(size=cfg['input_size'],
                     backbone_type=cfg['backbone_type'],
                     num_classes=cfg['num_classes'],
                     head_type=cfg['head_type'],
                     embd_shape=cfg['embd_shape'],
                     training=False)

# 3. 指定你那个 Loss 31.28 的权重文件路径
# 注意：路径指向你生成的那个 e_5.ckpt
ckpt_path = './checkpoints/arc_res50/e_5.ckpt'

if os.path.exists(ckpt_path + ".index"):
    print(f"[*] 正在加载权重: {ckpt_path}")
    model.load_weights(ckpt_path)

    # 4. 导出为成员 1 想要的 .h5 格式
    output_h5 = './checkpoints/arc_res50/arc_res50_final.h5'
    model.save_weights(output_h5)  # 只保存权重，成员1用起来最轻便

    print("-" * 50)
    print(f"祝贺！转换成功！")
    print(f"请将此文件发给成员1: {output_h5}")
    print("-" * 50)
else:
    print(f"[!] 错误：找不到权重文件，请检查路径是否正确。")