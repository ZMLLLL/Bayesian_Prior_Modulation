import logging
import time
import os
import shutil
import click
import torch
from utils.lr_scheduler import WarmupMultiStepLR
from net import Network
from tensorboardX import SummaryWriter


def create_logger(cfg):
    dataset = cfg.DATASET.DATASET
    net_type = cfg.BACKBONE.TYPE
    module_type = cfg.MODULE.TYPE
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_{}_{}.log".format(dataset, net_type, module_type, time_str)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file


def get_optimizer(cfg, model):
    base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})

    if cfg.TRAIN.OPTIMIZER.TYPE == "SGD":

        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
        # optimizer = torch.optim.SGD(
        #     [{'params': pre_params, 'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY_PRE},
        #      {'params': model.module.classifier.parameters(), 'weight_decay':cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY_POST}],
        #     lr=base_lr,
        #     momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
        #     # weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        #     nesterov=True,
        # )
    elif cfg.TRAIN.OPTIMIZER.TYPE == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        )
    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.TRAIN.LR_SCHEDULER.TYPE == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "cosine":
        if cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END, eta_min=1e-4
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.MAX_EPOCH, eta_min=1e-4
            )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
            warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARM_EPOCH,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler


def get_model(cfg, num_classes, device, logger):
    model = Network(cfg, mode="train", num_classes=num_classes)

    if cfg.BACKBONE.FREEZE == True:
        model.freeze_backbone()
        logger.info("Backbone has been freezed")

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # print(model.module.classifier)
    return model


def get_category_list(annotations, num_classes, cfg):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list


def save_info(cfg, logger, model, device):
    # close loop
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes")
    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboard")
        if cfg.TRAIN.TENSORBOARD.ENABLE
        else None
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        logger.info(
            "This directory has already existed, Please remember to modify your cfg.NAME"
        )
        if not click.confirm(
                "\033[1;31;40mContinue and override the former directory?\033[0m",
                default=False,
        ): exit(0)
        shutil.rmtree(code_dir)
        if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
    print("=> output model will be saved in {}".format(model_dir))
    this_dir = os.path.dirname(__file__)
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
    )
    shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)

    if tensorboard_dir is not None:
        # (1,3,cfg.INPUT_SIZE)
        dummy_input = torch.rand((1, 3) + cfg.INPUT_SIZE).to(device)
        # label_input = torch.rand((1)).to(device)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_graph(model if cfg.CPU_MODE else model.module, (dummy_input, ))
    else:
        writer = None

    return model_dir, writer