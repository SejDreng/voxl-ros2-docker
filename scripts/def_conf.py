import timm
cfg = timm.get_pretrained_cfg('mobilevitv2_100')
print(cfg.mean, cfg.std)