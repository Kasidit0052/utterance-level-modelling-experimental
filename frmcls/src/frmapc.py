import torch.nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

"""
Utterance-level Modelling Module:
Params:  hidden_size -> dimension of frame-level representation
         use_var     -> use a statistic pooling layer or not
         merge_mode  -> augment mode ["residual", "concat", "squeeze"]
         sq_ratio    -> reduction ratio for squeeze-and-excitation
         utterance_level_dropout ->  dropout on std before pooling
"""
class APCStatPoolingLayer(torch.nn.Module):
    def __init__(self, hidden_size, use_var = True, merge_mode="squeeze", sq_ratio = 2, utterance_level_dropout=None):
        super().__init__()

        self.num_hidden = hidden_size
        self.use_var = use_var
        self.merge_mode = merge_mode
        self.sq_ratio = sq_ratio
        self.utterance_level_dropout = utterance_level_dropout
        
        # Mode for using statistic pooling 
        if self.use_var:
            self.utterance_linear = torch.nn.Linear(2*self.num_hidden, self.num_hidden)

            # TODO[Experiment] regularization of utterance-level embeddings
            if self.utterance_level_dropout is not None:
                self.utterance_dropout = torch.nn.Dropout(self.utterance_level_dropout)

        # Mode for merging utterance level representation [concat, squeeze, residual]
        if self.merge_mode == "concat":
            self.frm_linear = torch.nn.Linear(2*self.num_hidden, self.num_hidden)
        if self.merge_mode == "squeeze":
            self.lin1 = torch.nn.Linear(self.num_hidden, self.num_hidden//self.sq_ratio)
            self.relu = torch.nn.ReLU()
            self.lin2 = torch.nn.Linear(self.num_hidden//self.sq_ratio, self.num_hidden)
            self.sigm = torch.nn.Sigmoid()

    def forward(self, feat, length, device):

        # Feat: Tensor(batch_size, max_len, mel_dims)
        pre_frm_embeddings = feat
        seq_length = length.unsqueeze(1)                  
        mean = torch.sum(feat,axis=1)*1/seq_length                          
        utt_mean = mean.unsqueeze(1)                                        # Tensor(batch_size, 1, mel_dims) 

        if self.use_var:
            # Hack for effectively computing variance for vary seq_len data
            frm_means = self.to_frame(utt_mean.squeeze(1), length, device)
            var = torch.sum((feat - frm_means)**2,dim = 1)*1/seq_length 
            std = torch.sqrt(var + 1e-9).unsqueeze(1)                       # Tensor(batch_size, 1, mel_dims)

            # TODO[Experiment] regularization of utterance-level embeddings (removing noisy dimension of mean or variance) [default:var]
            if self.utterance_level_dropout is not None:
                std = self.utterance_dropout(std)

            # Compute Utterance level embedding  
            concat_stat = torch.cat((utt_mean,std),2)                       # Tensor(batch_size, 1, mel_dims * 2)
            utt_embeddings = self.utterance_linear(concat_stat)             # Tensor(batch_size, 1, mel_dims)

        else:
            utt_embeddings = utt_mean  # Tensor(batch_size, 1, mel_dims)

        if self.merge_mode == "concat":
            # Concatenation + Linear layer
            frm_embeddings = self.to_frame(utt_embeddings.squeeze(1), length, device)
            frm_embeddings = torch.cat((feat,frm_embeddings),2) 
            frm_embeddings = self.frm_linear(frm_embeddings)

            frm_embeddings = pack_padded_sequence(frm_embeddings, length.cpu(), batch_first=True, enforce_sorted=False)
            frm_embeddings, _ = pad_packed_sequence(frm_embeddings, batch_first=True, total_length=int(length.max()))   
            return frm_embeddings, utt_embeddings, pre_frm_embeddings
        
        if self.merge_mode == "residual":
            # Residual Connection
            frm_embeddings = self.to_frame(utt_embeddings.squeeze(1), length, device)
            frm_embeddings += feat  
        if self.merge_mode == "squeeze":
            # Squeeze excitation connection
            squ = self.relu(self.lin1(utt_embeddings))
            exc = self.sigm(self.lin2(squ))
            frm_embeddings = feat * exc

        # removing zero from padded location in predictions
        post_frm_embeddings = pack_padded_sequence(frm_embeddings, length.cpu(), batch_first=True, enforce_sorted=False)
        post_frm_embeddings, _ = pad_packed_sequence(post_frm_embeddings, batch_first=True, total_length=int(length.max()))                                                            

        return post_frm_embeddings, utt_embeddings, pre_frm_embeddings
    
    def to_frame(self, utt_embeddings, length, device):
        padded_utt_embeddings  = torch.zeros([2 * len(length),utt_embeddings.shape[-1]]).to(device)
        padded_utt_embeddings[::2, ...] = utt_embeddings

        padded_length = torch.zeros([2 * len(length)], dtype=torch.int).to(device)
        padded_length[::2, ...]  =  length
        padded_length[1::2, ...] =  length.max().item() - length

        frm_embeddings = torch.repeat_interleave(padded_utt_embeddings, padded_length, dim=0)
        batch_size, max_len, frame_dim = len(length), int(length.max()), utt_embeddings.shape[-1]

        frm_embeddings = frm_embeddings.reshape(batch_size, max_len, frame_dim)

        ### Alternatives solution (very slower)
        # utt_embeddings = utt_embeddings.unsqueeze(1)
        # frm_list = [utt_embeddings[i].repeat(int(length[i].item()), 1) for i in range(0,len(length))] # List[Tensor(seq_len, mel_dims)]
        # frm_embeddings = torch.nn.utils.rnn.pad_sequence(frm_list, batch_first=True)                  # Tensor(batch_size, max_seq_len, mel_dims)
        return frm_embeddings
    
"""
LSTM + Utterance-level Modelling Downstream frame Classification:
Params:  input_size  -> dimension of input melspectogram
         hidden_size -> dimension of frame-level representation
         output_size -> number of phones
         num_layers  -> num LSTM layers
         dropout     -> LSTM dropout
         config_residual -> Residual Connection between LSTM layers
         use_var         -> use a statistic pooling layer or not
         merge_mode      -> augment mode ["residual", "concat", "squeeze"]
         sq_ratio        -> reduction ratio for squeeze-and-excitation
         use_preframe    -> use premodified feature if true
         utterance_level_dropout ->  dropout on std before pooling
         extract_layer   -> Depth where we want to extract feature
""" 
class UttAPCFrmLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0, 
                 config_residual=True, use_var=True, merge_mode="squeeze", sq_ratio = 2, 
                 use_preframe=False, utterance_level_dropout=None, extract_layer=3):
        super().__init__()

        input_sizes  = ([input_size] +[hidden_size] * (num_layers - 1))
        output_sizes = [hidden_size] * num_layers
        self.lstms = torch.nn.ModuleList([
            torch.nn.LSTM(in_size, out_size, dropout= dropout if i < len(input_sizes)-1 else 0.0, batch_first=True) 
            for i, (in_size, out_size) in enumerate(zip(input_sizes, output_sizes))
        ])
        self.pred = torch.nn.Linear(hidden_size, output_size)
        self.residual = config_residual
        self.use_preframe = use_preframe
        self.extract_layer = extract_layer

        assert self.extract_layer > 0 and self.extract_layer < len(self.lstms)+1

        self.utterance_level_dropout = utterance_level_dropout
        self.statpool = torch.nn.ModuleList([
            APCStatPoolingLayer(hidden_size, use_var, merge_mode, sq_ratio, self.utterance_level_dropout) for hidden_size in output_sizes
        ])

    def forward(self, feat, length, device):
        seq_len = feat.size(1)

        for i, layer in enumerate(self.lstms):
            
            packed_inputs  = pack_padded_sequence(feat, length.cpu(), batch_first=True, enforce_sorted=False) 
            packed_hidden,_ = layer(packed_inputs)
            seq_unpacked,_ = pad_packed_sequence(packed_hidden, batch_first=True, total_length=seq_len)   

            post_frm_embeddings, utt_embeddings, pre_frm_embeddings = self.statpool[i](seq_unpacked, length, device)
            
            if self.residual and feat.size(-1) == seq_unpacked.size(-1):
                post_frm_embeddings = post_frm_embeddings + feat
            feat = post_frm_embeddings

            if i+1 == self.extract_layer:
                if self.use_preframe:
                    return post_frm_embeddings, self.pred(pre_frm_embeddings), utt_embeddings, pre_frm_embeddings
                return post_frm_embeddings, self.pred(post_frm_embeddings), utt_embeddings, pre_frm_embeddings
    