# How to use scripts for training an frame-level classifer using original APC features
1. Training the model

    ```
    bash frmcls/train-old.sh path/to/train.conf start_epoch end_epoch
    ```

    Fit one epoch

    ```
    bash frmcls/train-old-epoch.sh path/to/train.conf epoch
    ```

2. Testing the model

    ```
    bash frmcls/test-old.sh path/to/test.conf start_epoch end_epoch
    ```

    Test one epoch

    ```
    bash frmcls/test-old-epoch.sh path/to/test.conf epoch
    ```

# Configuration for baseline APC

    1. Train Config

    ```
        {
            "feat_scp": "dataset/wsj/extra/si284-0.9-train.fbank.scp",
            "feat_mean_var": "dataset/wsj/extra/si284-0.9-train.mean-var",
            "label_scp": "dataset/wsj/extra/si284-0.9-train.bpali.scp",
            "label_set": "dataset/wsj/extra/phones.txt",
            "layers": 3,                       # Num LSTM Layers
            "hidden_size": 512,                # LSTM hidden dims
            "config_residual":true,            # Residual connections between LSTM
            "extract_layer":3,                 # Depth which we want to use the representation
            "seed":0,
            "apc_param": "path/to/APC/Param",  # Path to APC param
            "step_size": 0.001,
            "grad_clip": 5,
            "reduction": "mean",
            "optimizer": "adam",
            "batch_size": 32, 
            "step_accumulate": 1
        }
    ```

    2. Test Config

    ```
        {
            "label_set": "dataset/wsj/extra/phones.txt",
            "feat_scp": "dataset/wsj/extra/si284-0.9-dev.fbank.scp",
            "feat_mean_var": "dataset/wsj/extra/si284-0.9-train.mean-var",
            "layers": 3,                      # Num LSTM Layers
            "hidden_size": 512,               # LSTM hidden dims
            "config_residual":true,           # Residual connections between LSTM
            "extract_layer":3,                # Depth which we want to use the representation
            "apc_param": "path/to/APC/Param"  # Path to APC param
        }
    ```

# How to use scripts for training an frame-level classifer, using utterance-level modelling layer features
1. Training the model

    ```
    bash frmcls/train-utterance.sh path/to/train.conf start_epoch end_epoch
    ```

    Fit one epoch

    ```
    bash frmcls/train-utterance-epoch.sh path/to/train.conf epoch
    ```

2. Testing the model

    ```
    bash frmcls/test-utterance.sh path/to/test.conf start_epoch end_epoch
    ```

    Test one epoch

    ```
    bash frmcls/test-utterance-epoch.sh path/to/test.conf epoch
    ```

# Configuration for APC with utterance-level modelling

1. Train Config

    ```
        {
            "feat_scp": "dataset/wsj/extra/si284-0.9-train.fbank.scp",
            "feat_mean_var": "dataset/wsj/extra/si284-0.9-train.mean-var",
            "label_scp": "dataset/wsj/extra/si284-0.9-train.bpali.scp",
            "label_set": "dataset/wsj/extra/phones.txt",
            "layers": 3,                           # Num LSTM Layers
            "hidden_size": 512,                    # LSTM hidden dims
            "apc_param": "path/to/APC/Param",      # Path to APC param
            "step_size": 0.001,
            "grad_clip": 5, 
            "reduction": "mean",
            "optimizer": "adam",
            "batch_size": 32, 
            "step_accumulate": 1,
            "config_residual":false,               # Residual connections between LSTM
            "use_var":true,                        # Use a statistic Pooling Function
            "merge_mode":"residual",               # 3 possible augment function ["residual", "concat", "squeeze"]
            "sq_ratio": 2,                         # Reduction ratio for squeeze-and-excitation
            "seed": 0,
            "use_preframe":false,                  # If True: extracts pre-modified features
            "extract_layer":3                      # At what depth we want to extract features
        }
    ```

2. Extract Config

    ```
        {
            "label_set": "dataset/wsj/extra/phones.txt",
            "feat_scp": "dataset/wsj/extra/si284-0.9-dev.fbank.scp",
            "feat_mean_var": "dataset/wsj/extra/si284-0.9-train.mean-var",
            "layers": 3,                           # Num LSTM Layers
            "hidden_size": 512,                    # LSTM hidden dims
            "apc_param": "path/to/APC/Param",      # Path to APC param
            "config_residual":false,               # Residual connections between LSTM
            "use_var":true,                        # Use a statistic Pooling Function
            "merge_mode":"residual",               # 3 possible augment function ["residual", "concat", "squeeze"]
            "sq_ratio": 2,                         # Reduction ratio for squeeze-and-excitation
            "use_preframe":false,                  # If True: extracts pre-modified features
            "extract_layer":3                      # At what depth we want to extract features
        }
    ```

# Where could I find the implementation of frame-level probing classifier with utterance-level modelling layer ?
 
 ```
    frmcls/src/frmapc.py
 ```

