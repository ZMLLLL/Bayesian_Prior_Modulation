import _init_paths
from net import Network
from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import argparse
from core.evaluate import FusionMatrix, LossVector
from utils.utils import get_category_list
from loss import *


def parse_args():
    parser = argparse.ArgumentParser(description="BPM evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=True,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
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


def get_prior_estimate(para_dict_train, para_dict_test, LOSS_RATIO):
    device = para_dict_train["device"]

    # power type kernel
    num_class_list_train = para_dict_train["num_class_list"]
    num_class_list_train = [i ** LOSS_RATIO for i in num_class_list_train]
    prior_prob_train = num_class_list_train / np.sum(num_class_list_train)
    prior_prob_train = torch.FloatTensor(prior_prob_train).to(device)
    print("train dataset prior:")
    print(prior_prob_train)

    # exponential type kernel   
    # num_class_list_test = para_dict_test["num_class_list"]
    # prior_prob_test = num_class_list_test / np.sum(num_class_list_test)
    # prior_prob_test = torch.FloatTensor(prior_prob_test).to(device)
    # print("test dataset prior:")
    # print(prior_prob_test)
    # beta = 0.999
    # effective_num = [(1-beta**i)/(1-beta) for i in num_class_list_train]
    # print("effective number:")
    # print(effective_num)

    return prior_prob_train, None


def valid_model(dataLoader, model, num_classes, para_dict_train, para_dict_test,
                criterion, LOSS_RATIO):

    model.eval()

    top1_count, top3_count, top5_count, fusion_matrix, loss_vector = (
        [], [], [], FusionMatrix(num_classes), LossVector(num_classes))

    prior_prob_train, _ = get_prior_estimate(para_dict_train, para_dict_test, LOSS_RATIO)

    func = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (image, image_labels) in enumerate(dataLoader):
            image, image_labels = image.to(device), image_labels.to(device)
            output = model(image)

            # for j in range(len(image_labels)):
            #     loss = criterion(output[j:j+1,:], image_labels[j:j+1])
            #     loss_vector.update(loss.cpu().numpy(), image_labels[j].cpu().numpy())

            image_labels = image_labels.cpu().numpy()
            result = func(output)
            ######################################
            result = result / prior_prob_train
            ######################################
            _, top_k = result.topk(5, 1, True, True)
            score_result = result.cpu().numpy()
            fusion_matrix.update(score_result.argmax(axis=1), image_labels)

            topk_result = top_k.cpu().tolist()

            for i in range(len(image_labels)):
                top1_count += [topk_result[i][0] == image_labels[i]]
                top3_count += [image_labels[i] in topk_result[i][0:3]]
                top5_count += [image_labels[i] in topk_result[i][0:5]]

    top1_acc = float(np.sum(top1_count) / len(top1_count))
    top3_acc = float(np.sum(top3_count) / len(top1_count))
    top5_acc = float(np.sum(top5_count) / len(top1_count))
    print(
        "Top1:{:>5.2f}%  Top3:{:>5.2f}%  Top5:{:>5.2f}%".format(
            top1_acc * 100, top3_acc * 100, top5_acc * 100
        )
    )
    print("recall per class:")
    print(fusion_matrix.get_rec_per_class())
    print("precision per class:")
    print(fusion_matrix.get_pre_per_class())
    # print("loss per class:")
    # print(loss_vector.get_avg_loss_per_class())

    return top1_acc, fusion_matrix.get_rec_per_class()


if __name__ == "__main__":

    args = parse_args()
    update_config(cfg, args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print(cfg.DATASET.DATASET)

    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()

    annotations_train = train_set.get_annotations()
    num_class_list_train, cat_list_train = get_category_list(annotations_train, num_classes, cfg)
    annotations_test = test_set.get_annotations()
    num_class_list_test, cat_list_test = get_category_list(annotations_test, num_classes, cfg)

    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")

    para_dict_train = {
        "num_classes": num_classes,
        "num_class_list": num_class_list_train,
        "cfg": cfg,
        "device": device,
    }
    para_dict_test = {
        "num_classes": num_classes,
        "num_class_list": num_class_list_test,
        "cfg": cfg,
        "device": device,
    }

    criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict_train)

    model = Network(cfg, mode="test", num_classes=num_classes)
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
        # model_path = os.path.join(model_dir, "epoch_200.pth")

    print(model_path)

    model.load_model(model_path)
    print(model_path)

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    trainLoader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )


    if cfg.METHOD == "tau_norm":
        model_state_dict = model.state_dict()
        # set bias as zero
        model_state_dict['module.classifier.bias'].copy_(torch.zeros(
            (num_classes)))
        weight_ori = model_state_dict['module.classifier.weight']
        norm_weight = torch.norm(weight_ori, 2, 1)
        best_accuracy = 0
        best_p = 0
        for p in np.arange(0.0, 1.0, 0.1):
            ws = weight_ori.clone()
            for i in range(weight_ori.size(0)):
                ws[i] = ws[i] / torch.pow(norm_weight[i], p)
            model_state_dict['module.classifier.weight'].copy_(ws)
            print("\n___________________________", p, "__________________________________")
            acc, _ = valid_model(testLoader, model, num_classes, para_dict_train,
                                    para_dict_test,criterion, LOSS_RATIO=0)
            if acc > best_accuracy:
                best_accuracy = acc
                best_p = p
            print("when p is", best_p, ", best result is", best_accuracy)
    elif cfg.METHOD == "BPM":
        best_accuracy = 0
        best_LOSS_RATIO = 0
        for LOSS_RATIO in np.arange(0.0, 2.0, 0.1):
            print("\n___________________________", LOSS_RATIO, "__________________________________")
            acc, acc_per_class = valid_model(testLoader, model, num_classes, para_dict_train,
                                 para_dict_test, criterion, LOSS_RATIO)
            if acc > best_accuracy:
                best_accuracy = acc
                best_LOSS_RATIO = LOSS_RATIO
            print("when LOSS_RATIO is", best_LOSS_RATIO, ", best result is", best_accuracy)         

    else:
        print("Don not implement this method!")

     