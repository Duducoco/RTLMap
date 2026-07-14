#!/usr/bin/env python3
"""RTLMap 训练入口

用法示例:
    # 基础训练
    uv run python main.py --dataset-dir /path/to/dataset --data-root ./data

    # 启用 CodeBERT 文本编码
    uv run python main.py --dataset-dir /path/to/dataset --data-root ./data --use-text-encoder

    # 自定义超参数
    uv run python main.py --dataset-dir /path/to/dataset --data-root ./data \
        --max-epochs 200 --lr 3e-4 --batch-size 16 \
        --fusion-type ssm_film --precision bf16-mixed

    # 从检查点恢复训练
    uv run python main.py --dataset-dir /path/to/dataset --data-root ./data \
        --ckpt-path checkpoints/last.ckpt

    # 仅测试
    uv run python main.py --dataset-dir /path/to/dataset --data-root ./data \
        --test-only --ckpt-path checkpoints/best.ckpt
"""

import argparse
import logging
import sys

import torch

from datasets import DualGraphDataModule
from models.data_types import ModelConfig
from trainer.config import TrainerConfig
from trainer.app_config import AppConfig, DataConfig, RuntimeConfig
from trainer.lightning_module import DualGraphLightningModule
from trainer.lightning_trainer import LightningTrainer
from trainer.utils import train_model

# A100 等支持 Tensor Core 的 GPU 启用 TF32 加速
torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTLMap — 双图融合 GNN 覆盖率预测训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 数据 ──────────────────────────────────────────────
    data = p.add_argument_group("数据")
    data.add_argument(
        "--dataset-dir",
        type=str,
        nargs="+",
        required=True,
        help="coverage-report-extractor 输出目录（含 manifest.json），可传多个目录合并训练",
    )
    data.add_argument("--data-root", type=str, required=True, help="数据集缓存根目录")

    # ── 模型架构 ──────────────────────────────────────────
    model = p.add_argument_group("模型架构")
    model.add_argument("--hidden-dim", type=int, default=256)
    model.add_argument("--num-gnn-layers", type=int, default=6)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument(
        "--hyper-min-margin", type=float, default=0.01, help="超矩形每维度最小宽度"
    )
    model.add_argument(
        "--hyperrectangle-dim-per-type",
        type=int,
        default=10,
        help="每种 coverage type 的真实体积子矩形维度",
    )

    # ── 训练超参数 ────────────────────────────────────────
    train_g = p.add_argument_group("训练")
    train_g.add_argument("--max-epochs", type=int, default=100)
    train_g.add_argument("--lr", type=float, default=1e-4, help="学习率")
    train_g.add_argument("--weight-decay", type=float, default=1e-5)
    train_g.add_argument("--batch-size", type=int, default=16)
    train_g.add_argument("--num-workers", type=int, default=4)
    train_g.add_argument(
        "--use-bucketing",
        action="store_true",
        help="按图大小动态组 batch，减少 dense padding 和显存峰值",
    )
    train_g.add_argument(
        "--token-budget",
        type=int,
        default=8192,
        help="bucketing 模式下每个 batch 的 RTL+ASM 节点预算",
    )
    train_g.add_argument("--gradient-clip-val", type=float, default=1.0)
    train_g.add_argument("--accumulate-grad-batches", type=int, default=1)
    train_g.add_argument("--warmup-steps", type=int, default=100)
    train_g.add_argument(
        "--scheduler",
        choices=["cosine", "linear", "none"],
        default="cosine",
    )
    train_g.add_argument("--graph-loss-weight", type=float, default=1.0)
    train_g.add_argument(
        "--coverage-targets",
        nargs="+",
        default=["branch"],
        choices=["branch", "line", "fsm", "toggle", "condition"],
        help="用于图回归损失的覆盖率类型（默认仅 branch）",
    )

    # ── 设备与精度 ────────────────────────────────────────
    device = p.add_argument_group("设备")
    device.add_argument(
        "--accelerator",
        choices=["auto", "gpu", "cpu"],
        default="auto",
    )
    device.add_argument("--devices", default="auto", help="设备数量或 ID")
    device.add_argument(
        "--precision",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        default="bf16-mixed",
    )

    # ── 文本编码器 ────────────────────────────────────────
    text = p.add_argument_group("文本编码器 (CodeBERT)")
    text.add_argument(
        "--use-text-encoder",
        action="store_true",
        help="启用 CodeBERT 指令编码（默认回退到零向量）",
    )
    text.add_argument(
        "--text-model-name",
        default="microsoft/codebert-base",
        help="HuggingFace 模型名称",
    )
    text.add_argument("--text-output-dim", type=int, default=256)
    text.add_argument("--text-max-length", type=int, default=512)
    text.add_argument("--text-batch-size", type=int, default=1024)
    text.add_argument(
        "--text-pooling",
        choices=["mean", "cls"],
        default="mean",
    )

    # ── 实验管理 ──────────────────────────────────────────
    exp = p.add_argument_group("实验")
    exp.add_argument("--experiment-name", default="dual_graph_gnn")
    exp.add_argument(
        "--logger-type",
        choices=["tensorboard", "csv"],
        default="tensorboard",
    )
    exp.add_argument("--checkpoint-dir", default="checkpoints")
    exp.add_argument("--save-top-k", type=int, default=3)
    exp.add_argument("--early-stopping-patience", type=int, default=10)
    exp.add_argument("--log-every-n-steps", type=int, default=1)

    # ── 对比学习 (超矩形) ──────────────────────────────
    hyper = p.add_argument_group("对比学习 (Hyperrectangle)")
    hyper.add_argument(
        "--use-hyperrectangle", action="store_true", help="启用超矩形对比学习"
    )
    hyper.add_argument(
        "--contrastive-batch-size",
        type=int,
        default=16,
        help="对比 DataLoader batch 大小",
    )
    hyper.add_argument("--pair-candidate-pool-size", type=int, default=128)
    hyper.add_argument("--pair-relative-low-quota", type=int, default=2)
    hyper.add_argument("--pair-relative-mid-quota", type=int, default=1)
    hyper.add_argument("--pair-relative-high-quota", type=int, default=1)
    hyper.add_argument("--pair-sampling-seed", type=int, default=42)

    # ── coverage-vector pair 联合对比学习 (joint mode) ─────
    joint = p.add_argument_group("联合训练 (Joint Contrastive)")
    joint.add_argument(
        "--joint-contrastive",
        action="store_true",
        help="启用 coverage-vector pair 对比训练（覆盖率预测 + 对比学习）",
    )
    joint.add_argument(
        "--lambda-ce", type=float, default=1.0, help="a/b 两路监督损失的合并权重"
    )
    joint.add_argument(
        "--lambda-iou", type=float, default=1.0, help="逐类型真实体积 IoU 损失权重"
    )
    joint.add_argument(
        "--lambda-volume", type=float, default=0.25, help="真实体积校准损失权重"
    )
    joint.add_argument(
        "--volume-warmup-epochs",
        type=int,
        default=5,
        help="真实体积损失权重线性 warmup 的 epoch 数",
    )
    joint.add_argument(
        "--smooth-intersection-temperature",
        type=float,
        default=0.01,
        help="训练阶段平滑交集温度；验证和推理始终使用硬交集",
    )
    # ── 运行模式 ──────────────────────────────────────────
    mode = p.add_argument_group("运行模式")
    mode.add_argument("--ckpt-path", default=None, help="检查点路径（恢复训练或测试）")
    mode.add_argument("--test-only", action="store_true", help="仅运行测试")
    mode.add_argument(
        "--fast-dev-run", action="store_true", help="快速调试（单 batch）"
    )
    mode.add_argument("--seed", type=int, default=None, help="全局随机种子")

    return p.parse_args()


def build_model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        asm_instruction_dim=args.text_output_dim,
        use_hyperrectangle=args.use_hyperrectangle or args.joint_contrastive,
        hyper_min_margin=args.hyper_min_margin,
        hyperrectangle_dim_per_type=args.hyperrectangle_dim_per_type,
        coverage_target_keys=tuple(args.coverage_targets),
    )


def build_trainer_config(args: argparse.Namespace) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        scheduler_type=args.scheduler,
        graph_loss_weight=args.graph_loss_weight,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_workers=args.num_workers,
        use_bucketing=args.use_bucketing,
        token_budget=args.token_budget,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        save_top_k=args.save_top_k,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=args.log_every_n_steps,
        contrastive_batch_size=args.contrastive_batch_size,
        pair_candidate_pool_size=args.pair_candidate_pool_size,
        pair_relative_low_quota=args.pair_relative_low_quota,
        pair_relative_mid_quota=args.pair_relative_mid_quota,
        pair_relative_high_quota=args.pair_relative_high_quota,
        pair_sampling_seed=args.pair_sampling_seed,
        coverage_target_keys=tuple(args.coverage_targets),
        joint_contrastive=args.joint_contrastive,
        lambda_ce=args.lambda_ce,
        lambda_iou=args.lambda_iou,
        lambda_volume=args.lambda_volume,
        volume_warmup_epochs=args.volume_warmup_epochs,
        smooth_intersection_temperature=args.smooth_intersection_temperature,
    )


def build_text_encoder_config(args: argparse.Namespace):
    if not args.use_text_encoder:
        return None

    from text_encoder import TextEncoderConfig

    return TextEncoderConfig(
        model_name=args.text_model_name,
        output_dim=args.text_output_dim,
        max_length=args.text_max_length,
        batch_size=args.text_batch_size,
        pooling=args.text_pooling,
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.seed is not None:
        import lightning as L

        L.seed_everything(args.seed, workers=True)
        logger.info("全局随机种子: %d", args.seed)

    model_config = build_model_config(args)
    trainer_config = build_trainer_config(args)
    text_encoder_config = build_text_encoder_config(args)

    app_config = AppConfig(
        data=DataConfig(
            data_root=args.data_root,
            dataset_dir=args.dataset_dir,
        ),
        model=model_config,
        trainer=trainer_config,
        text_encoder=text_encoder_config,
        runtime=RuntimeConfig(
            logger_type=args.logger_type,
            experiment_name=args.experiment_name,
            ckpt_path=args.ckpt_path,
        ),
    )

    logger.info("实验: %s", args.experiment_name)
    logger.info(
        "模型: hidden_dim=%d, gnn_layers=%d, hyperrectangle=%s",
        model_config.hidden_dim,
        model_config.num_gnn_layers,
        model_config.use_hyperrectangle,
    )
    logger.info(
        "训练: epochs=%d, lr=%.1e, batch_size=%d",
        trainer_config.max_epochs,
        trainer_config.learning_rate,
        trainer_config.batch_size,
    )

    if args.test_only:
        if args.ckpt_path is None:
            logger.error("--test-only 需要指定 --ckpt-path")
            sys.exit(1)

        datamodule = DualGraphDataModule(
            root=args.data_root,
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            text_encoder_config=text_encoder_config,
        )
        datamodule.prepare_data()
        datamodule.setup("test")

        module = DualGraphLightningModule.load_from_checkpoint(
            args.ckpt_path,
            model_config=model_config,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            graph_loss_weight=args.graph_loss_weight,
            warmup_steps=args.warmup_steps,
            scheduler_type=args.scheduler,
            coverage_target_keys=tuple(args.coverage_targets),
            joint_contrastive=args.joint_contrastive,
            lambda_ce=args.lambda_ce,
            lambda_iou=args.lambda_iou,
            lambda_volume=args.lambda_volume,
            volume_warmup_epochs=args.volume_warmup_epochs,
            smooth_intersection_temperature=args.smooth_intersection_temperature,
        )
        lightning_trainer = LightningTrainer(
            trainer_config,
            experiment_name=args.experiment_name,
            logger_type=args.logger_type,
            has_validation=datamodule.val_dataloader() is not None,
        )
        results = lightning_trainer.test(module, datamodule, ckpt_path=args.ckpt_path)
        print(results)
        return

    train_model(app_config)


if __name__ == "__main__":
    main()
