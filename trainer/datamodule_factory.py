#!/usr/bin/env python3
"""训练 DataModule 组合工厂。"""

import lightning as L
from pathlib import Path

from datasets import DualGraphDataModule
from datasets.pair_datamodule import ContrastivePairDataModule


class JointTrainDataModule(L.LightningDataModule):
    """训练 dataloader 来自 pair loader，验证/测试来自普通 DataModule。"""

    def __init__(
        self,
        base: DualGraphDataModule,
        pair_dm: ContrastivePairDataModule,
        pair_val_dm: ContrastivePairDataModule,
    ):
        super().__init__()
        self._base = base
        self._pair_dm = pair_dm
        self._pair_val_dm = pair_val_dm

    def prepare_data(self):
        self._base.prepare_data()

    def setup(self, stage=None):
        self._base.setup(stage)
        if stage == "fit" or stage is None:
            self._pair_dm.set_sample_indices(self._base.train_indices)
            self._pair_val_dm.set_sample_indices(self._base.val_indices or [])
            self._pair_dm.setup(stage)
            self._pair_val_dm.setup(stage)

    def train_dataloader(self):
        return self._pair_dm.train_dataloader()

    def val_dataloader(self):
        base_loader = self._base.val_dataloader()
        if len(self._pair_val_dm) == 0:
            return base_loader
        return [base_loader, self._pair_val_dm.val_dataloader()]

    def test_dataloader(self):
        return self._base.test_dataloader()


def build_training_datamodule(
    *,
    data_root: str,
    dataset_dir,
    text_encoder_config,
    trainer_config,
) -> tuple[L.LightningDataModule, bool]:
    """构建训练 DataModule，并返回是否具备验证集。"""
    if trainer_config.joint_contrastive:
        pair_dm = ContrastivePairDataModule(
            dataset_dir=dataset_dir,
            batch_size=trainer_config.contrastive_batch_size,
            num_workers=min(trainer_config.num_workers, 4),
            pairs_per_sample=trainer_config.contrastive_pairs_per_sample,
            text_encoder_config=text_encoder_config,
            processed_root=str(Path(data_root) / "train"),
            asm_chunk_files=64,
        )
        pair_val_dm = ContrastivePairDataModule(
            dataset_dir=dataset_dir,
            batch_size=trainer_config.contrastive_batch_size,
            num_workers=min(trainer_config.num_workers, 4),
            shuffle=False,
            pairs_per_sample=trainer_config.contrastive_pairs_per_sample,
            text_encoder_config=text_encoder_config,
            processed_root=str(Path(data_root) / "train"),
            asm_chunk_files=64,
        )
        base_dm = DualGraphDataModule(
            root=data_root,
            dataset_dir=dataset_dir,
            text_encoder_config=text_encoder_config,
            batch_size=trainer_config.batch_size,
            num_workers=trainer_config.num_workers,
            asm_chunk_files=64,
            use_bucketing=trainer_config.use_bucketing,
            token_budget=trainer_config.token_budget,
        )
        return JointTrainDataModule(base_dm, pair_dm, pair_val_dm), True

    return (
        DualGraphDataModule(
            root=data_root,
            dataset_dir=dataset_dir,
            text_encoder_config=text_encoder_config,
            batch_size=trainer_config.batch_size,
            num_workers=trainer_config.num_workers,
            asm_chunk_files=64,
            use_bucketing=trainer_config.use_bucketing,
            token_budget=trainer_config.token_budget,
        ),
        True,
    )
