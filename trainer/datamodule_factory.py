#!/usr/bin/env python3
"""训练 DataModule 组合工厂。"""

import lightning as L

from datasets import DualGraphDataModule
from datasets.manifest import is_contrastive_dataset_dir, normalize_dataset_dirs
from datasets.triple_datamodule import ContrastiveTripleDataModule


class JointTrainDataModule(L.LightningDataModule):
    """训练 dataloader 来自 triple loader，验证/测试来自普通 DataModule。"""

    def __init__(
        self,
        base: DualGraphDataModule,
        triple_dm: ContrastiveTripleDataModule,
    ):
        super().__init__()
        self._base = base
        self._triple_dm = triple_dm

    def prepare_data(self):
        self._base.prepare_data()

    def setup(self, stage=None):
        self._base.setup(stage)
        self._triple_dm.setup(stage)

    def train_dataloader(self):
        return self._triple_dm.train_dataloader()

    def val_dataloader(self):
        return self._base.val_dataloader()

    def test_dataloader(self):
        return self._base.test_dataloader()


class TripleTrainOnlyDataModule(L.LightningDataModule):
    """仅使用 contrastive 三元组训练集的数据模块。"""

    def __init__(self, triple_dm: ContrastiveTripleDataModule):
        super().__init__()
        self._triple_dm = triple_dm

    def setup(self, stage=None):
        self._triple_dm.setup(stage)

    def train_dataloader(self):
        return self._triple_dm.train_dataloader()

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        return []


def build_training_datamodule(
    *,
    data_root: str,
    dataset_dir,
    text_encoder_config,
    trainer_config,
) -> tuple[L.LightningDataModule, bool]:
    """构建训练 DataModule，并返回是否具备验证集。"""
    dataset_dirs = normalize_dataset_dirs(dataset_dir)
    is_contrastive_only = (
        trainer_config.joint_contrastive
        and dataset_dirs
        and all(is_contrastive_dataset_dir(p) for p in dataset_dirs)
    )

    if trainer_config.joint_contrastive:
        triple_dm = ContrastiveTripleDataModule(
            dataset_dir=dataset_dir,
            index_file=trainer_config.contrastive_triple_index or None,
            batch_size=trainer_config.contrastive_batch_size,
            num_workers=min(trainer_config.num_workers, 4),
        )
        if is_contrastive_only:
            return TripleTrainOnlyDataModule(triple_dm), False

        base_dm = DualGraphDataModule(
            root=data_root,
            dataset_dir=dataset_dir,
            text_encoder_config=text_encoder_config,
            batch_size=trainer_config.batch_size,
            num_workers=trainer_config.num_workers,
            use_bucketing=trainer_config.use_bucketing,
            token_budget=trainer_config.token_budget,
        )
        return JointTrainDataModule(base_dm, triple_dm), True

    return (
        DualGraphDataModule(
            root=data_root,
            dataset_dir=dataset_dir,
            text_encoder_config=text_encoder_config,
            batch_size=trainer_config.batch_size,
            num_workers=trainer_config.num_workers,
            use_bucketing=trainer_config.use_bucketing,
            token_budget=trainer_config.token_budget,
        ),
        True,
    )
