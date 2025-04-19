import oyaml as yaml
import os
import torch

def create_ckpt_dir():
    ckpt_dir = "./ckpt"
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
        os.mkdir(os.path.join(ckpt_dir, "val_vis"))
        os.mkdir(os.path.join(ckpt_dir, "models"))
    return ckpt_dir


def to_items(dic):
    return dict(map(_to_item, dic.items()))


def _to_item(item):
    return item[0], item[1].item()


class Config(dict):
    def __init__(self, conf_file):
        with open(conf_file, "r") as f:
            config = yaml.safe_load(f)
        self._conf = config

    def __getattr__(self, name):
        return self._conf.get(name, None)

    def __getitem__(self, key):
        return self._conf[key]


def conf_to_param(config: dict) -> dict:
    dind_keys = []
    rm_keys = []
    for key, val in config.items():
        if isinstance(val, dict):
            dind_keys.append(key)
        elif not isinstance(val, (float, int, bool, str)):
            rm_keys.append(key)  # bug was: using .pop(key) on a list

    for target in dind_keys:
        val = config.pop(target)
        config.update(val)
    for target in rm_keys:
        del config[target]

    return config


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device("cpu")
    state_dict = obj.state_dict()
    return {k: v.to(cpu_device) for k, v in state_dict.items()}


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {"n_iter": n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()

    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name, map_location="cpu")
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict["n_iter"]
