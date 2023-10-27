# How to use scripts for training an a baseline APC
1. Training the model

    ```
    bash apc/train-old.sh path/to/train.conf start_epoch end_epoch
    ```

    Fit one epoch

    ```
    bash apc/train-old-epoch.sh path/to/train.conf epoch
    ```

2. Extracting the utterance-level embeddings

    ```
    bash apc/extract-old.sh path/to/extract.conf epoch
    ```

# Configuration for baseline APC

1. Train Config

    ```
    {
    "feat_scp": "dataset/librispeech/extra/train-clean-360.fbank.scp",
    "feat_mean_var": "dataset/librispeech/extra/train-clean-360.fbank.mean-var",
    "time_shift": 5,                  # APC timeshift(K) params
    "layers": 3,                      # Num LSTM Layers
    "hidden_size": 512,               # LSTM hidden dims
    "step_size": 0.001,
    "grad_clip": 5, 
    "reduction": "mean",
    "optimizer": "adam",
    "batch_size": 32, 
    "step_accumulate": 1,
    "config_residual":false,          # Residual connections between LSTM
    "seed": 0
    }
    ```

2. Extract Config

    ```
    {
        "feat_scp": "dataset/librispeech/extra/dev-clean.fbank.scp",
        "feat_mean_var": "dataset/librispeech/extra/train-clean-360.fbank.mean-var",
        "time_shift": 5,                  # APC timeshift(K) param
        "layers": 3,                      # Num LSTM Layers
        "hidden_size": 512,               # LSTM hidden dims
        "batch_size": 32,
        "config_residual":false,          # Residual connections between LSTM
        "extract_layer":3                 # Layer that we want to extract utterance-level embeddings
    }
    ```

# How to train an APC with utterance-level modelling

1. Training the model

    ```
    bash apc/train-utterance.sh path/to/train.conf start_epoch end_epoch
    ```

    Fit one epoch

    ```
    bash apc/train-utterance-epoch.sh path/to/train.conf epoch
    ```

2. Extracting the utterance-level embeddings

    ```
    bash apc/extract-utterance-.sh path/to/extract.conf epoch
    ```

# Configuration for APC with utterance-level modelling

1. Train Config

    ```
    {
    "feat_scp": "dataset/librispeech/extra/train-clean-360.fbank.scp",
    "feat_mean_var": "dataset/librispeech/extra/train-clean-360.fbank.mean-var",
    "time_shift": 5,                    # APC timeshift(K) param
    "layers": 3,                        # Num LSTM Layers
    "hidden_size": 512,                 # LSTM hidden dims
    "step_size": 0.001,
    "grad_clip": 5, 
    "reduction": "mean",
    "optimizer": "adam",
    "batch_size": 32, 
    "step_accumulate": 1,
    "config_residual":false,            # Residual connections between LSTM
    "use_var":true,                     # Use a statistic Pooling Function
    "merge_mode":"residual",            # 3 possible augment function ["residual", "concat", "squeeze"]
    "sq_ratio": 2,                      # Reduction ratio for squeeze-and-excitation
    "seed": 0
    }

    ```

2. Extract Config

    ```
    {
        "feat_scp": "dataset/librispeech/extra/dev-clean.fbank.scp",
        "feat_mean_var": "dataset/librispeech/extra/train-clean-360.fbank.mean-var",
        "time_shift": 5,                    # APC timeshift(K) param
        "layers": 3,                        # Num LSTM Layers
        "hidden_size": 512,                 # LSTM hidden dims
        "batch_size": 32, 
        "config_residual":false,            # Residual connections between LSTM
        "use_var":true,                     # Use a statistic Pooling Function
        "merge_mode":"residual",            # 3 possible augment function ["residual", "concat", "squeeze"]
        "sq_ratio": 2,                      # Reduction ratio for squeeze-and-excitation 
        "extract_layer": 2
    }
    ```

# Where could I find the implementation of utterance-level modelling layer ?
 
 ```
    apc/src/uttapc.py
    apc/src/uttapc_extractor.py
 ```


