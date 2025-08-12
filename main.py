from argparse import ArgumentParser
import pandas as pd
from utils import get_dataloader, collate_fn, compute_metrics
from transformers import ViTForImageClassification, Trainer, TrainingArguments


def train(args):
    meta_data = pd.read_csv('meta/metadata.csv')
    train_data, val_data, test_data = get_dataloader(
        meta_data,
        args.data_root,
    )

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=183,
        ignore_mismatched_sizes=True
    )
    for param in model.vit.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",  # evaluate more frequently if desired
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,  # save checkpoint every 500 steps
        load_best_model_at_end=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        report_to="tensorboard",
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/malte/datasets/FungiImages')
    parser.add_argument('--output_dir', type=str, default='./vit-base-checkpoints')
    parser.add_argument('--logging_dir', type=str, default='./logs')

    # hyperparams
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args()
    train(args)
