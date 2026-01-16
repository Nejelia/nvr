import os,yaml
cfg = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
masks_dir = cfg["paths"]["masks_dir"]
print("masks_dir (as in config):", masks_dir)
print("abs path:", os.path.abspath(masks_dir))
print("exists:", os.path.isdir(masks_dir))
if os.path.isdir(masks_dir):
    print("listing:", sorted(os.listdir(masks_dir)))
else:
    print("NO DIR")
