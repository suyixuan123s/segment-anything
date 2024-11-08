# import torch
# from box import Box
# from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.Model import Model
#
# import lightning as L
# from config import cfg
# fabric = L.Fabric(accelerator="auto",
#                       devices=cfg.num_devices,
#                       strategy="auto",
#                       loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
# fabric.launch()
# fabric.seed_everything(1337 + fabric.global_rank)
#
#
#
# with fabric.device:
#     model = Model(cfg)
#     model.setup()
# train_data = HaNDataset(cfg)
# train_loader = DataLoader(train_data, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
# train_data = fabric._setup_dataloader(train_loader)
# def configure_opt(cfg: Box, model: Model):
#     def lr_lambda(step):
#         if step < cfg.opt.warmup_steps:
#             return step / cfg.opt.warmup_steps
#         elif step < cfg.opt.steps[0]:
#             return 1.0
#         elif step < cfg.opt.steps[1]:
#             return 1 / cfg.opt.decay_factor
#         else:
#             return 1 / (cfg.opt.decay_factor**2)
#     optimizer
# = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#     return optimizer, scheduler
# optimizer, scheduler = configure_opt(cfg, model)
# model, optimizer = fabric.setup(model, optimizer)
