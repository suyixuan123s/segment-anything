from box import Box
config = {
    "num_devices": 1,
    "batch_size": 6,
    "num_workers": 4,
    "num_epochs": 20,
    "save_interval": 2,
    "resume": None,
    "out_dir": "模型权重输出地址",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "SAM的权重地址",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
    "dataset": {
        "root_dir": "数据集的根目录",
        "sample_num": 4,
        "target_size": 1024
    }
}
cfg = Box(config)
