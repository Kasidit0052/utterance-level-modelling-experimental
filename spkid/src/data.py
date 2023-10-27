import dataprep


class Vox1Spkid:
    def __init__(self, feat_scp, feat_mean=None, feat_var=None, shuffling=False, rand=None):
        f = open(feat_scp)
        self.feat_entries = [dataprep.parse_scp_line(line) for line in f.readlines()]
        f.close()

        self.feat_mean = feat_mean
        self.feat_var = feat_var

        self.indices = list(range(len(self.feat_entries)))

        if shuffling:
            rand.shuffle(self.indices)

    def __iter__(self):
        for i in self.indices:
            feat_key, feat_file, feat_shift = self.feat_entries[i]
            feat = dataprep.load_feat(feat_file, feat_shift, self.feat_mean, self.feat_var)
            spk = feat_key[:7]

            yield feat_key, spk, feat

