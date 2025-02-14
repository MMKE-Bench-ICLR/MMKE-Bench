import os
from argparse import ArgumentParser
from pprint import pprint

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.models.ke_train import MLLM_KE

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dirpath",
        type=str,
        default="KE/models/",  
        help="The address where the model is saved",
    )
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_type", type=str, required=True, help="The entity name that the model holds")  

    parser = MLLM_KE.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args, _ = parser.parse_known_args()
    pprint(args.__dict__)

    args.gpus = -1
    seed_everything(seed=args.seed)

    args.train_data_path = f"data_json/{args.data_type}_train.json"
    args.dev_data_path = f"data_json/{args.data_type}_eval.json"

    model_dirpath = os.path.join(args.dirpath, args.data_type, args.model_name)
    logger = TensorBoardLogger(model_dirpath, name=None)

    callbacks = [
        ModelCheckpoint(
            monitor="valid_acc",
            mode="max",
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            save_top_k=args.save_top_k,
            filename="model-{epoch:02d}-{valid_acc:.4f}",
        ),
        LearningRateMonitor(
            logging_interval="step",
        ),
    ]

    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
    
    model = MLLM_KE(**vars(args))

    trainer.fit(model)
