import _init_paths
from loss import *
from dataset import *
from config import cfg, update_config
from utils.utils import (create_logger, get_optimizer, get_scheduler,
                         get_model, get_category_list, save_info)
from core.function import train_model, valid_model
import torch
import os
from torch.utils.data import DataLoader
import argparse
import warnings
import torch.backends.cudnn as cudnn
import ast


def parse_args():
    parser = argparse.ArgumentParser(description="codes for BBN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type=ast.literal_eval,
        dest='auto_resume',
        required=False,
        default=True,
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--gpu_ids",
        help="decide which gpu to use",
        required=False,
        default="0",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    logger, log_file = create_logger(cfg)
    warnings.filterwarnings("ignore")
    cudnn.benchmark = True
    auto_resume = args.auto_resume

    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)

    annotations = train_set.get_annotations()
    num_classes = train_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    num_class_list, cat_list = get_category_list(annotations, num_classes, cfg)
    # print(num_class_list)
    # print(cat_list)

    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "cfg": cfg,
        "device": device,
    }
    criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)
    epoch_number = cfg.TRAIN.MAX_EPOCH

    # ----- BEGIN MODEL BUILDER -----
    model = get_model(cfg, num_classes, device, logger)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    # ----- END MODEL BUILDER -----
    trainLoader = DataLoader(train_set,
                             batch_size=cfg.TRAIN.BATCH_SIZE,
                             shuffle=cfg.TRAIN.SHUFFLE,
                             num_workers=cfg.TRAIN.NUM_WORKERS,
                             pin_memory=cfg.PIN_MEMORY,
                             drop_last=True)
    validLoader = DataLoader(
        valid_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    # ----- SAVE INFORMATION ---------
    model_dir, writer = save_info(cfg, logger, model, device)

    best_result, best_epoch, start_epoch = 0, 0, 1

    # ----- BEGIN RESUME ---------
    all_models = os.listdir(model_dir)
    if len(all_models) <= 1 or auto_resume == False:
        auto_resume = False
    else:
        all_models.remove("best_model.pth")
        resume_epoch = max(
            [int(name.split(".")[0].split("_")[-1]) for name in all_models])
        resume_model_path = os.path.join(model_dir,
                                         "epoch_{}.pth".format(resume_epoch))

    if cfg.RESUME_MODEL != "" or auto_resume:
        if cfg.RESUME_MODEL == "":
            resume_model = resume_model_path
        else:
            resume_model = cfg.RESUME_MODEL if '/' in cfg.RESUME_MODEL else os.path.join(
                model_dir, cfg.RESUME_MODEL)
        logger.info("Loading checkpoint from {}...".format(resume_model))
        checkpoint = torch.load(resume_model,
                                map_location="cpu" if cfg.CPU_MODE else "cuda")
        if cfg.CPU_MODE:
            model.load_model(resume_model)
        else:
            model.module.load_model(resume_model)
        if cfg.RESUME_MODE != "state_dict":
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_result = checkpoint['best_result']
            best_epoch = checkpoint['best_epoch']

    logger.info(
        "-------------------Train start :{}  {}-------------------".format(
            cfg.BACKBONE.TYPE, cfg.MODULE.TYPE))

    for epoch in range(start_epoch, epoch_number + 1):

        scheduler.step()

        # if epoch>=180:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.00005

        lr = next(iter(optimizer.param_groups))['lr']
        print("learning rate is ", lr)

        train_acc, train_loss = train_model(trainLoader, model, epoch,
                                            epoch_number, device, optimizer,
                                            criterion, cfg, logger)
        model_save_path = os.path.join(
            model_dir,
            "epoch_{}.pth".format(epoch),
        )
        if epoch % cfg.SAVE_STEP == 0:
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, model_save_path)

        loss_dict, acc_dict = {
            "train_loss": train_loss
        }, {
            "train_acc": train_acc
        }

        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            valid_acc, valid_loss = valid_model(para_dict, validLoader, epoch,
                                                model, cfg, criterion, logger)

            loss_dict["valid_loss"], acc_dict[
                "valid_acc"] = valid_loss, valid_acc
            if valid_acc > best_result:
                best_result, best_epoch = valid_acc, epoch
                torch.save(
                    {
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_result,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join(model_dir, "best_model.pth"))
            logger.info(
                "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------"
                .format(best_epoch, best_result * 100))
        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars("scalar/acc", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)

    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()
    logger.info(
        "-------------------Train Finished :{}-------------------".format(
            cfg.NAME))
