from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Sequence

import torch
import torch.multiprocessing as mp

from .core.dist import (
    get_global_rank,
    get_local_rank,
    get_master_ip,
    get_world_size,
)
from .core.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E2FGVI")
    parser.add_argument(
        "-c",
        "--config",
        default=str(Path(__file__).resolve().parent / "configs/train_e2fgvi.json"),
        type=str,
    )
    parser.add_argument("-p", "--port", default="23455", type=str)
    return parser


def main_worker(rank: int, config: dict) -> None:
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank
    if config['distributed']:
        torch.cuda.set_device(int(config['local_rank']))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=config['init_method'],
                                             world_size=config['world_size'],
                                             rank=config['global_rank'],
                                             group_name='mtorch')
        print('using GPU {}-{} for training'.format(int(config['global_rank']),
                                                    int(config['local_rank'])))

    config['save_dir'] = os.path.join(
        config['save_dir'],
    '{}_{}'.format(config['model']['net'],
               os.path.basename(config['config_path']).split('.')[0]))

    config['save_metric_dir'] = os.path.join(
        './scores',
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else:
        config['device'] = 'cpu'

    if (not config['distributed']) or config['global_rank'] == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['save_metric_dir'], exist_ok=True)
        config_path = os.path.join(config['save_dir'],
                                   os.path.basename(config['config_path']))
        if not os.path.isfile(config_path):
            copyfile(config['config_path'], config_path)
        print('[**] create folder {}'.format(config['save_dir']))

    trainer = Trainer(config)
    trainer.train()


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    torch.backends.cudnn.benchmark = True
    mp.set_sharing_strategy('file_system')

    config_path = Path(args.config).expanduser().resolve()
    with open(config_path, 'r', encoding='utf-8') as fp:
        config = json.load(fp)

    config['config_path'] = str(config_path)
    config['world_size'] = get_world_size()
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"
    config['distributed'] = config['world_size'] > 1
    print(config['world_size'])

    if get_master_ip() == "127.0.0.1":
        mp.spawn(main_worker, nprocs=config['world_size'], args=(config, ))
    else:
        config['local_rank'] = get_local_rank()
        config['global_rank'] = get_global_rank()
        main_worker(-1, config)


if __name__ == "__main__":
    main()
