import torch.nn as nn
import torch.nn.functional as F
import torch
from xfkd.weight import constant_init, kaiming_init
from nets.DAN import DAN
from nets.cbam import CBAM, ChannelAttention, SpatialAttention
class XFKD_FeatureLoss(nn.Module):
    """
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma (float, optional): Weight of mask_loss. Defaults to 0.001
        llambda (float, optional): Weight of relation_loss. Defaults to 0.000005
    """

    def __init__(self,
                 student_channels, teacher_channels, name,temp=0.5, alpha=0.001, beta=0.0005, gamma=0.001, llambda=0.000005):
        super(XFKD_FeatureLoss, self).__init__()
        self.name = name
        self.temp = temp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.llambda = llambda

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))

        self.bn_s = nn.BatchNorm2d(student_channels)
        self.bn_t = nn.BatchNorm2d(teacher_channels)
        self.dan = DAN(teacher_channels)

        self.reset_parameters()

    def forward(self, preds_S, preds_T, gt_bboxes, img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        preds_S = torch.add(preds_T, preds_S)

        preds_S = self.bn_s(preds_S)
        preds_T = self.bn_t(preds_T)

        preds_Ses = preds_S.split(1, dim=0)
        preds_Tes = preds_T.split(1, dim=0)
        img_metas = img_metas
        new_preds_Ses, new_preds_Tes, new_img_metas, new_gt_bboxes = [], [], [], []

        for idx, gt_bboxe in enumerate(gt_bboxes):
            if gt_bboxe.size(0) != 0:
                new_preds_Ses.append(preds_Ses[idx])
                new_preds_Tes.append(preds_Tes[idx])
                new_img_metas.append(img_metas[idx])
                new_gt_bboxes.append(gt_bboxe)
        if (len(new_preds_Ses) == 0):
            return 0
        preds_S = torch.cat(new_preds_Ses, dim=0)
        preds_T = torch.cat(new_preds_Tes, dim=0)
        img_metas = new_img_metas
        gt_bboxes = new_gt_bboxes

        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)

        N, C, H, W = preds_S.shape

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)
        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            # if gt_bboxes[i].size(0) == 0:
            #     print(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i][1] * W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i][1] * W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i][0] * H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i][0] * H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / \
                   (wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
                    torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i] > 0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg,
                                             C_attention_s, C_attention_t,
                                             S_attention_s, S_attention_t)
        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)


        fgd_loss = self.alpha * fg_loss + self.beta * bg_loss + self.gamma * mask_loss + self.llambda * rela_loss
        #
        return fgd_loss

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W = preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        spatial_map = SpatialAttention(value).mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((spatial_map / temp).view(N, -1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = ChannelAttention(value).mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
        C_attention = C * F.softmax(channel_map / temp, dim=1)  # Ac _mask

        return S_attention, C_attention

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        mask_loss = F.smooth_l1_loss(torch.add(C_t,C_s),C_t) + F.smooth_l1_loss(torch.add(S_t,S_s),S_t)
        return mask_loss

    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        loss_bce = nn.BCELoss(reduction='mean')
        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        out_s1 = self.dan(preds_S)
        out_t1 = self.dan(preds_T)

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s  # # Rs(F)
        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t  # # Rc(F)
        rela_loss1 = loss_mse(out_s, out_t) / len(out_s)

        out_s1 = F.softmax(out_s1, 2).detach()
        out_t1 = F.softmax(out_t1, 2).detach()
        rela_loss2 = loss_bce(out_s1, out_t1) / len(out_s1)

        rela_loss = rela_loss1 + rela_loss2

        return rela_loss

    def constant_init(module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True
        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)
