import torch
import numpy as np
import data as utils
from torch.utils import data 

class WSJDataset(data.Dataset):
    def __init__(self, feat_scp, label_scp, feat_mean=None, feat_var=None, label_dict=None, shuffling=False, rand=None):
        f = open(feat_scp)
        self.feat_entries = [utils.parse_scp_line(line) for line in f.readlines()]
        f.close()

        f = open(label_scp)
        self.label_entries = [utils.parse_scp_line(line) for line in f.readlines()]
        f.close()

        assert len(feat_scp) == len(label_scp)

        self.feat_mean = feat_mean
        self.feat_var = feat_var
        self.label_dict = label_dict

        self.indices = list(range(len(self.feat_entries)))

        if shuffling:
            rand.shuffle(self.indices)

    def __len__(self):
        return len(self.label_entries)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]  # type: ignore
        else:
            feat_key, feat_file, feat_shift = self.feat_entries[self.indices[index]]
            feat = utils.load_feat(feat_file, feat_shift, self.feat_mean, self.feat_var)

            label_key, label_file, label_shift = self.label_entries[self.indices[index]]
            labels = utils.load_labels(label_file, label_shift)
            length = len(feat)

            assert feat_key == label_key
            return feat_key, feat, labels, length 
        
    def one_hot(self, labels):
        label_dict = self.label_dict
        result = np.zeros((len(labels), len(label_dict)))
        for i, ell in enumerate(labels):
            result[i, label_dict[ell]] = 1
        return result

    def collate_fn(self, batch):
        key    = [item[0] for item in batch]
        feat   = [torch.Tensor(item[1]) for item in batch]
        length = torch.Tensor([item[3] for item in batch])
        labels = [item[2] for item in batch]
        one_hots = [torch.Tensor(self.one_hot(item[2])) for item in batch]

        return [key, feat, length, labels, one_hots]

        
