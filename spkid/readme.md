# How to use scripts for training an utterance-level classifer using original APC features
1. Training the model

    ```
    bash spkid/train-probe.sh path/to/train.conf start_epoch end_epoch
    ```

    Fit one epoch

    ```
    bash spkid/train-probe-epoch.sh path/to/train.conf epoch
    ```

2. Testing the model

    ```
    bash frmcls/predict-probe.sh path/to/test.conf start_epoch end_epoch
    ```

    Test one epoch

    ```
    bash frmcls/predict-epoch.sh path/to/test.conf epoch
    ```

# Configuration for baseline APC

    1. Train Config

    ```
        {
            "spk_set": "dataset/vox1/extra/speakers.txt",
            "feat_scp": "dataset/vox1/extra/vox1-spkid-train.fbank.scp",
            "feat_mean_var": "dataset/vox1/extra/vox1-spkid-train.fbank.mean-var",
            "hidden_size": 512,                     # LSTM hidden dims                    
            "layers": 3,                            # Num LSTM Layers
            "step_size": 0.001,
            "grad_clip": 5,
            "apc_param":  "path/to/APC/Param",      # Path to APC param
            "config_residual":false,                # Residual connections between LSTM
            "extract_layer":3,                      # At what depth we want to extract features
            "reduction": "mean",
            "optimizer": "adam",
            "batch_size": 16, 
            "step_accumulate": 1,
            "seed":0
        }
    ```

    2. Test Config

    ```
        {
            "spk_set": "dataset/vox1/extra/speakers.txt",
            "feat_scp": "dataset/vox1/extra/vox1-spkid-train.fbank.scp",
            "feat_mean_var": "dataset/vox1/extra/vox1-spkid-train.fbank.mean-var",
            "layers": 3,                          # Num LSTM Layers
            "hidden_size": 512,                   # LSTM hidden dims             
            "config_residual":false,              # Residual connections between LSTM
            "extract_layer":3,                    # At what depth we want to extract features
            "apc_param": "path/to/APC/Param"      # Path to APC param
        }
    ```

# How to use scripts for training an utterance-level classifer, using utterance-level modelling layer features

1. Training the model

    ```
    bash spkid/train-probe-utterance.sh path/to/train.conf start_epoch end_epoch
    ```

    Fit one epoch

    ```
    bash spkid/train-probe-utterance-epoch.sh path/to/train.conf epoch
    ```

2. Testing the model

    ```
    bash spkid/predict-probe-utterance.sh path/to/test.conf start_epoch end_epoch
    ```

    Test one epoch

    ```
    bash spkid/predict-utterance-epoch.sh path/to/test.conf epoch
    ```

# Configuration for APC with utterance-level modelling

1. Train Config

    ```
        {
            "spk_set": "dataset/vox1/extra/speakers.txt",
            "feat_scp": "dataset/vox1/extra/vox1-spkid-train.fbank.scp",
            "feat_mean_var": "dataset/vox1/extra/vox1-spkid-train.fbank.mean-var",
            "hidden_size": 512,                   # LSTM hidden dims     
            "layers": 3,                          # Num LSTM Layers
            "step_size": 0.001,
            "grad_clip": 5,
            "apc_param":  "path/to/APC/Param",    # Path to APC param
            "reduction": "mean",
            "optimizer": "adam",
            "batch_size": 16 , 
            "step_accumulate": 1,
            "config_residual":false,              # Residual connections between LSTM
            "use_var":true,                       # Use a statistic Pooling Function
            "merge_mode":"residual",              # 3 possible augment function ["residual", "concat", "squeeze"]
            "sq_ratio": 2,                        # Reduction ratio for squeeze-and-excitation
            "extract_mode":"post-frame",          # 2 possible extract location ["postframe","utterance-vector"]
            "extract_layer":3,                    # At what depth we want to extract features
            "seed": 0
        }
    ```

2. Extract Config

    ```
        {
            "spk_set": "dataset/vox1/extra/speakers.txt",
            "feat_scp": "dataset/vox1/extra/vox1-spkid-train.fbank.scp",
            "feat_mean_var": "dataset/vox1/extra/vox1-spkid-train.fbank.mean-var",
            "hidden_size": 512,                   # LSTM hidden dims     
            "layers": 3,                          # Num LSTM Layers
            "apc_param":  "path/to/APC/Param",    # Path to APC param
            "batch_size": 16,
            "config_residual":false,              # Residual connections between LSTM
            "use_var":true,                       # Use a statistic Pooling Function
            "merge_mode":"residual",              # 3 possible augment function ["residual", "concat", "squeeze"]
            "sq_ratio": 2,                        # Reduction ratio for squeeze-and-excitation
            "extract_mode":"post-frame",          # 2 possible extract location ["postframe","utterance-vector"]
            "extract_layer":3,                    # At what depth we want to extract features
        }

    ```

# Where could I find the implementation of frame-level probing classifier with utterance-level modelling layer ?
 
 ```
    spkid/src/spkapc.py
 ```