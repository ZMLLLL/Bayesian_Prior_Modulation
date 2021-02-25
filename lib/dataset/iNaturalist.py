from dataset.baseset import BaseSet
import random


class iNaturalist(BaseSet):
    def __init__(self, mode='train', cfg=None, transform=None):
        super().__init__(mode, cfg, transform)
        random.seed(0)
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and mode == "train":
            self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
            self.class_dict = self._get_class_dict()

    def __getitem__(self, index):
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.mode == 'train':
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.transform(img)

        if self.mode != 'test':
            image_label = now_info['category_id']  # 0-index

        return image, image_label
