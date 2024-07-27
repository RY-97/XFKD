# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.teacher import TYoloBody
from nets.student import SYoloBody
from nets.yolo_training import YOLOLoss, weights_init
from util.callbacks import LossHistory, EvalCallback
from util.dataloader import YoloDataset, yolo_dataset_collate
from util.utils import get_anchors, get_classes
from util.utils_fit_one import fit_one_epoch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # --------------------------------------------------------#
    #   训练前一定要修改classes_path，使其对应自己的数据集
    # --------------------------------------------------------#
    classes_path = 'model_data/SIX-ray_classes.txt'
    # ---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    # ---------------------------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # ------------------------------------------------------#
    #   输入的shape大小，一定要是32的倍数
    # ------------------------------------------------------#
    input_shape = [416, 416]

    model_path_student = 'model_data/Epoch1-loss51.5270-val_loss8.3291.pth'   #加载学生模型

    model_path_teacher = 'model_data/best_epoch_weights-T.pth'     #加载教师模型

    # -------------------------------#
    #   所使用的主干特征提取网络
    # -------------------------------#
    backbone_s = "mobilenetv3"

    backbone_t = "cspdarknet53"

    pretrained = False
    # ------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    # ------------------------------------------------------#
    mosaic = False
    Cosine_lr = False
    label_smoothing = 0.005

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 4
    Freeze_lr = 1e-3
    # ----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    # ----------------------------------------------------#
    UnFreeze_Epoch = 200
    Unfreeze_batch_size = 4
    Unfreeze_lr = 1e-4
    # ------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------#
    Freeze_Train = True

    save_dir = 'logs'
    save_period = 10

    eval_flag = True
    eval_period = 10

    # ------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------#
    num_workers = 8
    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    setup_seed(1234)
    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    model = SYoloBody(anchors_mask, num_classes, backbone=backbone_s, pretrained=pretrained)
    model_teacher = TYoloBody(anchors_mask, num_classes, backbone=backbone_t, pretrained=pretrained)

    if not pretrained:
        weights_init(model)
    if model_path_student != '':
        print('Load weights {}.'.format(model_path_student))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path_student, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if model_path_teacher != '':
        print('Load weights teacher model:{}.'.format(model_path_teacher))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model_teacher.state_dict()
        pretrained_dict = torch.load(model_path_teacher, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model_teacher.load_state_dict(model_dict)

    model_train = model.train()
    model_teacher = model_teacher.eval()
    # model_teacher = model_teacher.train()
    if Cuda:
        model_train = model
        model_teacher_train = model_teacher
        cudnn.benchmark = True
        model_train = model_train.cuda()
        model_teacher_train = model_teacher_train.cuda()

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)

    time_str = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=0)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines,
                                     log_dir, Cuda, eval_flag=eval_flag, period=eval_period)

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, model_teacher_train, yolo_loss, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, save_period, save_dir)
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines,
                                     log_dir, Cuda, eval_flag=eval_flag, period=eval_period)

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, model_teacher_train, yolo_loss, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, save_period, save_dir)
            lr_scheduler.step()
