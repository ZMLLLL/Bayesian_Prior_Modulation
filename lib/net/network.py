import torch
import torch.nn as nn
from backbone import res50,res32_cifar
from aug_module import MLP_AUG, Conv_AUG
from modules import GAP, Identity, FCNorm, DistFC
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, cfg, mode="train", num_classes=1000):
        super(Network, self).__init__()
        pretrain = (
            True
            if mode == "train"
               and cfg.RESUME_MODEL == ""
               and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.num_classes = num_classes
        self.cfg = cfg

        self.backbone = eval(self.cfg.BACKBONE.TYPE)(
            self.cfg,
            pretrain=pretrain,
            pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
            last_layer_stride=2,
        )

        if cfg.MPL_AUG.ENABLE is True:
            self.aug_module = MLP_AUG(cfg)
        elif cfg.CON_AUG.ENABLE is True:
            self.aug_module = Conv_AUG(cfg)
        else:
            self.aug_module = None

        self.module = self._get_module()
        self.classifier = self._get_classifer()
        self.feature_len = self.get_feature_length()

    def forward(self, image, noise):
        ###########################################
        x = self.backbone(image)
        x = self.module(x)
        rep_feature = x.view(x.shape[0], -1)
        ###########################################
        if not noise is None:
            if self.cfg.MPL_AUG.ENABLE:
                aug_feature = self.aug_module(noise + rep_feature)
            elif self.cfg.CON_AUG.ENABLE:
                aug_feature = self.aug_module(image + noise)
                aug_feature = self.module(aug_feature)
                aug_feature = aug_feature.view(aug_feature.shape[0], -1)
            feat_all = torch.cat((rep_feature, aug_feature), 0)
        else:
            aug_feature = None
            feat_all = rep_feature
        ###########################################

        output = self.classifier(feat_all)
        return output, feat_all
        # return x 


    def freeze_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone has been loaded...")

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Model has been loaded...")

    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64
        else:
            num_features = 2048
        return num_features

    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module = Identity()
        else:
            raise NotImplementedError
        return module

    def _get_classifer(self):
        bias_flag = self.cfg.CLASSIFIER.BIAS
        num_features = self.get_feature_length()
        if self.cfg.CLASSIFIER.TYPE == "FCNorm":
            classifier = FCNorm(num_features, self.num_classes)
        elif self.cfg.CLASSIFIER.TYPE == "FC":
            # classifier = FCNorm(num_features, self.num_classes)
            classifier = nn.Linear(num_features, self.num_classes, bias=bias_flag)
        elif self.cfg.CLASSIFIER.TYPE == "DistFC":
            classifier = DistFC(num_features, self.num_classes)
        else:
            raise NotImplementedError

        return classifier









    
