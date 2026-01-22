#!/usr/bin/env python3
"""
Parse training log txt to extract train/val loss/acc arrays and print as lists.
Usage: run and input the txt path when prompted.
"""
import os
import re
import sys


def parse_log(file_path: str):
    pattern = re.compile(
        r"Epoch\s+\[\d+/\d+\],\s*"
        r"Train Loss:\s*([0-9.]+),\s*"
        r"Train Acc:\s*([0-9.]+),\s*"
        r"Val Loss:\s*([0-9.]+),\s*"
        r"Val Acc:\s*([0-9.]+)")

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped.startswith("Epoch"):
                continue
            m = pattern.search(stripped)
            if not m:
                continue
            tl, ta, vl, va = map(float, m.groups())
            train_loss.append(tl)
            train_acc.append(ta)
            val_loss.append(vl)
            val_acc.append(va)

    return train_acc, val_acc, train_loss, val_loss


def format_series(name: str, values, width: int) -> str:
    vals_str = ", ".join(f"{v:.4f}" for v in values)
    padding = " " * max(0, width - len(name))
    return f"{name} = {padding}[{vals_str}];"


def main():
    file_path = input("请输入日志txt文件路径: ").strip()
    if not file_path:
        print("未提供路径，退出")
        sys.exit(1)
    if not os.path.isfile(file_path):
        print(f"文件不存在: {file_path}")
        sys.exit(1)

    train_acc, val_acc, train_loss, val_loss = parse_log(file_path)
    if not train_acc:
        print("未在文件中找到任何Epoch行，检查格式是否正确")
        sys.exit(1)

    max_name = max(len("train_acc"), len("val_acc"), len("train_loss"), len("val_loss"))

    print(format_series("train_acc", train_acc, max_name))
    print(format_series("val_acc",   val_acc,   max_name))
    print(format_series("train_loss", train_loss, max_name))
    print(format_series("val_loss",   val_loss,   max_name))


if __name__ == "__main__":
    main()
