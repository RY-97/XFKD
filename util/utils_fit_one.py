import os
import torch
from tqdm import tqdm
from util.utils import get_lr


def fit_one_epoch(model_train, model, model_teacher_train, yolo_loss, loss_history, eval_callback,optimizer,
                  epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, save_period, save_dir):

    loss = 0
    val_loss = 0

    print('\nStart Train')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    model_teacher_train.eval()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets, img_shapes, annotation_lines = batch[0], batch[1], batch[2], batch[3]
        with torch.no_grad():
            if cuda:
                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
            else:
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        # ----------------------#
        #   前向传播
        # ----------------------#
        with torch.no_grad():
            outputs_teacher = model_teacher_train(images)
        outputs, fgd_loss, mgd_loss = model_train(images, outputs_teacher, targets, img_shapes)
        loss_value_all = 0  # =0
        num_pos_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
        for l in range(len(outputs)):
            loss_item, num_pos = yolo_loss(l, outputs[l], targets)
            loss_value_all += loss_item
            num_pos_all += num_pos
        loss_value = (loss_value_all + fgd_loss + mgd_loss) / num_pos_all  # 无sum

        # ----------------------#
        #   反向传播
        # ----------------------#
        loss_value.backward()
        optimizer.step()
        loss += loss_value.item()

        pbar.set_postfix(**{'loss': loss / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)

    pbar.close()
    print('Finish Train')


    print('\nStart Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, img_shapes, annotation_lines = batch[0], batch[1], batch[2], batch[3]
        with torch.no_grad():
            if cuda:
                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
            else:
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
                # ----------------------#
                #   前向传播
                # ----------------------#
            outputs = model_train(images)

            loss_value_all = 0  # =0
            num_pos_all = 0
                # ----------------------#
                #   计算损失
                # ----------------------#
            for l in range(len(outputs)):
                loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
                num_pos_all += num_pos
            loss_value = loss_value_all / num_pos_all
            val_loss += loss_value.item()

        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
        pbar.update(1)

    pbar.close()
    print('Finish Validation')

    loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    eval_callback.on_epoch_end(epoch + 1, model_train)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
        epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))