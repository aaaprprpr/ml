from ultralytics import YOLO
import torch.multiprocessing as mp
import torch


if __name__ == '__main__':
    print("当前设备:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    # model = YOLO('./yolo-pose.yaml')
    model = YOLO('./mobilenet.yaml')
    model.to('cuda') 
    model.train(data='./data.yaml', epochs=400, batch=32, imgsz=640,
            #lr0= 0.01, # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            #lrf= 0.01, # (float) final learning rate (lr0 * lrf)
            #optimizer='AdamW',             # 优化器选择，默认为 'SGD'
        #     close_mosaic=10,             # 关闭Mosaic增强的epoch数，默认为 10
            rect=False,                  # 是否使用矩形训练，默认为 False
            box=8,                    # 目标检测损失权重，默认为 0.05
            cls=0.5,                     # 分类损失权重，默认为 0.5
            dfl= 2,                   #1.5
            pose=1.0,                     # 关键点损失权重?
            # cls_pw=1.0,                  # 分类正样本权重，默认为 1.0
            # obj=1.0,                     # 对象性损失权重，默认为 1.0
            # obj_pw=1.0,                  # 对象性正样本权重，默认为 1.0
            # iou_t=0.20,                  # IoU阈值，默认为 0.20
            # anchor_t=4.0,                # 锚点阈值，默认为 4.0
            # fl_gamma=0.0,                # Focal Loss gamma，默认为 0.0
            hsv_h=0.015,                 # HSV-Hue增广强度，默认为 0.015色调
            hsv_s=0.7,                   # HSV-Saturation增广强度，默认为 0.7饱和度
            hsv_v=0.4,                   # HSV-Value增广强度，默认为 0.4
            degrees=100,                 # 图像旋转角度，默认为 0.0
            # translate=0.1,               # 图像平移强度，默认为 0.1
            scale=0.2,                   # 图像缩放强度，默认为 0.5
            # shear=0.0001,                   # 图像剪切强度，默认为 0.0
            perspective=0.0001,             # 图像透视变换强度，默认为 0.0
            # mosaic=1.0,                  # Mosaic增强概率，默认为 1.0
        #     mixup=0.01,                   # MixUp增强概率，默认为 0.0MixUp通过线性插值两张图像及其标签生成新的训练样本，增加数据多样性并有助于平滑决策边界。
        #     copy_paste=0.2,              # Copy-Paste增强概率，默认为 0.0)  # train  
            patience=50,    #早停
    )