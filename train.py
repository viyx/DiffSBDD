import argparse
from argparse import Namespace
from pathlib import Path
import warnings

import pytorch_lightning as pl
import wandb
import yaml
import numpy as np

from lightning_modules import ARCLigandPocketDDPM


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)

    args = p.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args = merge_args_and_yaml(args, config)

    if args.wandb_params.run_id and not args.resume:
        raise ValueError('Cannot use runid explicitly without resuming.')

    if args.wandb_params.mode != 'disabled':
        Path.mkdir(Path(args.logdir, 'wandb'), exist_ok=True, parents=True)

    run_id = args.wandb_params.run_id or wandb.util.generate_id()
    out_dir = Path(args.logdir, args.run_name, run_id)

    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()
    pl_module = ARCLigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation
    )

    ckpt_filename = 'last'
    ckpt_dir = Path(out_dir, 'checkpoints')

    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project='ligand-pocket-ddpm',
        group=args.run_name,
        # name=args.run_name,
        id=run_id,
        resume='must' if args.resume else None,
        tags=[args.wandb_params.tags],
        # entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
        log_model='all',
        # checkpoint_name=ckpt_filename + '.ckpt'
    )



    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_filename,
        verbose=True,
        # filename="last-model-epoch={epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
        # every_n_epochs=1,
    )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accelerator=args.acc
    )

    ckpt_path = None
    if args.resume:
        ckpt_fullpath = Path(checkpoint_callback.dirpath, ckpt_filename + '.ckpt')

        # local ckpt
        if Path.exists(ckpt_fullpath):
            ckpt_path = ckpt_fullpath

        # wandb ckpt
        elif logger.experiment.resumed:
            art = logger.experiment.use_artifact(f"model-{run_id}:latest", type="model")
            wandb_ckpt_dir = art.download()
            ckpt_path = Path(wandb_ckpt_dir, 'model.ckpt')
        else:
            raise FileNotFoundError("No checkpoint to resume. Consider `resume`: False")

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)
