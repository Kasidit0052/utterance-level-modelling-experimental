import torch
import numpy as np
import dataprep
from torch.utils import data 

class Vox1Dataset(data.Dataset):
    def __init__(self, feat_scp, feat_mean=None, feat_var=None, spk_set=None, shuffling=False, rand=None):

        f = open(feat_scp)
        self.feat_entries = [dataprep.parse_scp_line(line) for line in f.readlines()]
        f.close()

        self.feat_mean = feat_mean
        self.feat_var = feat_var
        self.spk_dict = dataprep.load_label_dict(spk_set)

        #TODO(Change shuffling so that it work on APC and frmcls)
        if shuffling:
            rand.shuffle(self.feat_entries)
    
    def __len__(self):
        return len(self.feat_entries)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]  # type: ignore
        else:
            feat_key, feat_file, feat_shift = self.feat_entries[index]
            feat = dataprep.load_feat(feat_file, feat_shift, self.feat_mean, self.feat_var)
            length = len(feat)
            spk = feat_key[:7]
            return feat_key, feat, spk, length 
        
    def one_hot(self, spk, spk_dict):
        result = np.zeros(len(spk_dict))
        result[spk_dict[spk]] = 1
        return result

    def collate_fn(self, batch):
        key    = [item[0] for item in batch]
        feat   = [torch.Tensor(item[1]) for item in batch]
        length = torch.Tensor([item[3] for item in batch])
        labels = [item[2] for item in batch]
        one_hots = torch.Tensor(np.array([self.one_hot(item[2], self.spk_dict) for item in batch]))
        return [key, feat, length, labels, one_hots]
