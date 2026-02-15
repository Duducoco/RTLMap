#!/usr/bin/env python3
"""RTLMap 训练入口

用法示例:
    # 基础训练
    python main.py --data-root ./data

    # 指定模块和仿真目录
    python main.py --data-root ./data \
        --sim-results-dirs ./sim_a ./sim_b \
        --module-names alu decoder

    # 启用 CodeBERT 文本编码
    python main.py --data-root ./data --use-text-encoder

    # 自定义超参数
    python main.py --data-root ./data \
        --max-epochs 200 --lr 3e-4 --batch-size 16 \
        --fusion-type ssm_film --precision bf16-mixed

    # 从检查点恢复训练
    python main.py --data-root ./data --ckpt-path checkpoints/last.ckpt

    # 仅测试
    python main.py --data-root ./data --test-only --ckpt-path checkpoints/best.ckpt
"""

import argparse
import logging
import sys

from datasets import DualGraphDataModule
from models.data_types import ModelConfig
from trainer.config import TrainerConfig
from trainer.lightning_module import DualGraphLightningModule
from trainer.lightning_trainer import LightningTrainer
from trainer.utils import train_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTLMap — 双图融合 GNN 覆盖率预测训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 数据 ──────────────────────────────────────────────
    data = p.add_argument_group("数据")
    data.add_argument("--data-root", type=str, required=True, help="数据集根目录")
    data.add_argument(
        "--sim-results-dirs", nargs="*", default=None,
        help="simulation_results 目录路径列表",
    )
    data.add_argument(
        "--module-names", nargs="*", default=None,
        help="要标注的模块名列表（不含 .json 后缀）",
    )

    # ── 模型架构 ──────────────────────────────────────────
    model = p.add_argument_group("模型架构")
    model.add_argument("--hidden-dim", type=int, default=256)
    model.add_argument("--num-gnn-layers", type=int, default=4)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument(
        "--fusion-type", choices=["film", "ssm_film"], default="ssm_film",
        help="融合模式",
    )
    model.add_argument("--ssm-d-state", type=int, default=16, help="SSM 状态空间维度")
    model.add_argument(
        "--ssm-pool-mode", choices=["last", "mean", "attention"], default="last",
    )

    # ── 训练超参数 ────────────────────────────────────────
    train_g = p.add_argument_group("训练")
    train_g.add_argument("--max-epochs", type=int, default=100)
    train_g.add_argument("--lr", type=float, default=1e-4, help="学习率")
    train_g.add_argument("--weight-decay", type=float, default=1e-5)
    train_g.add_argument("--batch-size", type=int, default=32)
    train_g.add_argument("--num-workers", type=int, default=4)
    train_g.add_argument("--gradient-clip-val", type=float, default=1.0)
    train_g.add_argument("--accumulate-grad-batches", type=int, default=1)
    train_g.add_argument("--warmup-steps", type=int, default=100)
    train_g.add_argument(
        "--scheduler", choices=["cosine", "linear", "none"], default="cosine",
    )
    train_g.add_argument("--edge-loss-weight", type=float, default=1.0)
    train_g.add_argument("--graph-loss-weight", type=float, default=1.0)
    train_g.add_argument("--label-smoothing", type=float, default=0.1)

    # ── 设备与精度 ────────────────────────────────────────
    device = p.add_argument_group("设备")
    device.add_argument(
        "--accelerator", choices=["auto", "gpu", "cpu"], default="auto",
    )
    device.add_argument("--devices", default="auto", help="设备数量或 ID")
    device.add_argument(
        "--precision",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        default="32-true",
    )

    # ── 文本编码器 ────────────────────────────────────────
    text = p.add_argument_group("文本编码器 (CodeBERT)")
    text.add_argument(
        "--use-text-encoder", action="store_true",
        help="启用 CodeBERT 指令编码（默认回退到零向量）",
    )
    text.add_argument(
        "--text-model-name", default="microsoft/codebert-base",
        help="HuggingFace 模型名称",
    )
    text.add_argument("--text-output-dim", type=int, default=256)
    text.add_argument("--text-max-length", type=int, default=512)
    text.add_argument("--text-batch-size", type=int, default=256)
    text.add_argument(
        "--text-pooling", choices=["mean", "cls"], default="mean",
    )

    # ── 实验管理 ──────────────────────────────────────────
    exp = p.add_argument_group("实验")
    exp.add_argument("--experiment-name", default="dual_graph_gnn")
    exp.add_argument(
        "--logger-type", choices=["tensorboard", "csv"], default="tensorboard",
    )
    exp.add_argument("--checkpoint-dir", default="checkpoints")
    exp.add_argument("--save-top-k", type=int, default=3)
    exp.add_argument("--early-stopping-patience", type=int, default=10)
    exp.add_argument("--log-every-n-steps", type=int, default=10)

    # ── 运行模式 ──────────────────────────────────────────
    mode = p.add_argument_group("运行模式")
    mode.add_argument("--ckpt-path", default=None, help="检查点路径（恢复训练或测试）")
    mode.add_argument("--test-only", action="store_true", help="仅运行测试")
    mode.add_argument("--fast-dev-run", action="store_true", help="快速调试（单 batch）")
    mode.add_argument("--seed", type=int, default=None, help="全局随机种子")

    return p.parse_args()


def build_model_config(args: argparse.Namespace) -> ModelConfig:
    """从 CLI 参数构建 ModelConfig"""
    return ModelConfig(
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        fusion_type=args.fusion_type,
        ssm_d_state=args.ssm_d_state,
        ssm_pool_mode=args.ssm_pool_mode,
        asm_instruction_dim=args.text_output_dim,
    )


def build_trainer_config(args: argparse.Namespace) -> TrainerConfig:
    """从 CLI 参数构建 TrainerConfig"""
    return TrainerConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        scheduler_type=args.scheduler,
        edge_loss_weight=args.edge_loss_weight,
        graph_loss_weight=args.graph_loss_weight,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        save_top_k=args.save_top_k,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=args.log_every_n_steps,
    )


def build_text_encoder_config(args: argparse.Namespace):
    """启用文本编码器时构建 TextEncoderConfig，否则返回 None"""
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

    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 全局随机种子
    if args.seed is not None:
        import lightning as L

        L.seed_everything(args.seed, workers=True)
        logger.info("全局随机种子: %d", args.seed)

    # 构建配置
    model_config = build_model_config(args)
    trainer_config = build_trainer_config(args)
    text_encoder_config = build_text_encoder_config(args)

    logger.info("实验: %s", args.experiment_name)
    logger.info("模型: hidden_dim=%d, gnn_layers=%d, fusion=%s",
                model_config.hidden_dim, model_config.num_gnn_layers,
                model_config.fusion_type)
    logger.info("训练: epochs=%d, lr=%.1e, batch_size=%d",
                trainer_config.max_epochs, trainer_config.learning_rate,
                trainer_config.batch_size)

    if args.test_only:
        # 仅测试模式
        if args.ckpt_path is None:
            logger.error("--test-only 需要指定 --ckpt-path")
            sys.exit(1)

        module = DualGraphLightningModule.load_from_checkpoint(
            args.ckpt_path, model_config=model_config,
        )
        datamodule = DualGraphDataModule(
            root=args.data_root,
            sim_results_dirs=args.sim_results_dirs,
            module_names=args.module_names,
            text_encoder_config=text_encoder_config,
            batch_size=trainer_config.batch_size,
            num_workers=trainer_config.num_workers,
        )
        lt = LightningTrainer(
            config=trainer_config,
            experiment_name=args.experiment_name,
            logger_type=args.logger_type,
            has_validation=False,
        )
        lt.test(module, datamodule)
    else:
        # 训练模式（可选从检查点恢复）
        module, trainer = train_model(
            model_config=model_config,
            trainer_config=trainer_config,
            data_root=args.data_root,
            sim_results_dirs=args.sim_results_dirs,
            module_names=args.module_names,
            text_encoder_config=text_encoder_config,
            has_validation=True,
            has_test=False,
            logger_type=args.logger_type,
            experiment_name=args.experiment_name,
            ckpt_path=args.ckpt_path,
        )
        logger.info("训练完成")


if __name__ == "__main__":
    main()
