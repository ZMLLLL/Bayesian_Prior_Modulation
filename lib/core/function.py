import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix
import numpy as np
import torch
import time
import torch.nn.functional as F
from loss import regularization


def train_model(
    trainLoader,model,epoch,epoch_number,device,optimizer,criterion,cfg,logger,**kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    start_time = time.time()
    number_batch = len(trainLoader)
    func = torch.nn.Softmax(dim=1)
    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label) in enumerate(trainLoader):

        cnt = label.shape[0]
        image, label = image.to(device), label.to(device)
        output = model(image)
        loss = criterion(output, label)
        now_result = torch.argmax(func(output), 1)

        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)

    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    logger.info(pbar_str)
    return acc.avg, all_loss.avg


def valid_model(
    para_dict, dataLoader, epoch_number, model, cfg, criterion, logger):

    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)

    num_class_list = para_dict["num_class_list"]
    device = para_dict["device"]
    num_class_list = [i**cfg.LOSS.RATIO for i in num_class_list]
    prior_prob = num_class_list / np.sum(num_class_list)
    prior_prob = torch.FloatTensor(prior_prob).to(device)


    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (image, label) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            output = model(image)
            loss = criterion(output, label)
            score_result = func(output)
            score_result = score_result / prior_prob
            now_result = torch.argmax(score_result, 1)

            all_loss.update(loss.data.item(), label.shape[0])

            fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())

            now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc.update(now_acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc.avg * 100
        )
        logger.info(pbar_str)
        # print(fusion_matrix.get_rec_per_class())
        # print(fusion_matrix.get_pre_per_class())
    return acc.avg, all_loss.avg
