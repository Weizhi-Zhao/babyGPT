from omegaconf import OmegaConf, DictConfig, ListConfig
from pathlib import Path
import pickle
import logging
logger = logging.getLogger(__name__)

def recursive_load_config(cfg):
    if isinstance(cfg, DictConfig):
        temp_cfg = OmegaConf.create()
        for k, v in cfg.items():
            if isinstance(v, str) and Path(v).suffix == '.yaml':
                if not Path(v).exists():
                    logger.warning(f"Config file {v} does not exist")
                    continue
                sub_cfg = OmegaConf.load(v)
                sub_cfg = recursive_load_config(sub_cfg)
                temp_cfg = OmegaConf.merge(temp_cfg, sub_cfg)
            elif isinstance(v, (DictConfig, ListConfig)):
                cfg[k] = recursive_load_config(v)
        cfg = OmegaConf.merge(cfg, temp_cfg)
    elif isinstance(cfg, ListConfig):
        pass
        # temp_cfg = cfg.copy()
        # for i, item in enumerate(cfg):
        #     if isinstance(item, str) and Path(item).exists() and Path(item).suffix == '.yaml':
        #         sub_cfg = OmegaConf.load(item)
        #         sub_cfg = recursive_load_config(sub_cfg)
        #         temp_cfg = OmegaConf.merge(temp_cfg, sub_cfg)
        #     elif isinstance(item, (DictConfig, ListConfig)):
        #         cfg[i] = recursive_load_config(item)
    else:
        raise ValueError(f"Unsupported type: {type(cfg)}, "
                         "only DictConfig and ListConfig are supported")
    return cfg


def load_config(config_path):
    cfg = OmegaConf.load(config_path)
    cfg = recursive_load_config(cfg)
    return cfg

def get_vocab_size(path: Path | str):
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == '.yaml':
        path = load_config(path).meta_path
    elif path.suffix == '.pkl':
        pass

    with open(path, 'rb') as f:
        meta = pickle.load(f)

    return meta['vocab_size']

if __name__ == '__main__':
    # Example usage
    # cfg = load_config('configs/shakespeare_char.yaml')
    # print(OmegaConf.to_yaml(cfg))
    print(get_vocab_size('configs/shakespeare_char.yaml'))