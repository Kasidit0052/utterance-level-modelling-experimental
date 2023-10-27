import torch
import data as utils
from torch.utils import data 

def collate_fn(batch):
    key    = [item[0] for item in batch]
    feat   = [torch.Tensor(item[1]) for item in batch]
    length = torch.Tensor([item[2] for item in batch])
    return [key, feat, length]

### Libispeech dataset ###
class LibriSpeechDataset(data.Dataset):
    def __init__(self, feat_scp, feat_mean=None, feat_var=None, shuffling=False, rand=None):
        f = open(feat_scp)
        self.feat_entries = [utils.parse_scp_line(line) for line in f.readlines()]
        f.close()

        self.feat_mean = feat_mean
        self.feat_var = feat_var

        if shuffling:
            rand.shuffle(self.feat_entries)

    def __len__(self):
        return len(self.feat_entries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]  # type: ignore
        else:
            key, file, shift = self.feat_entries[index]
            feat = utils.load_feat(file, shift, self.feat_mean, self.feat_var)
            length = len(feat)
            return key, feat, length

