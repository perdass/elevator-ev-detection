"""
电梯内电动车检测 - 训练脚本
"""
from ultralytics import YOLO
import argparse
import os

def train(args):
    # 加载预训练模型
    model = YOLO(args.model)
    
    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        verbose=True,
        seed=42,
        deterministic=True,
    )
    
    print(f"\n训练完成！结果保存在: runs/detect/{args.name}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练电梯电动车检测模型")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="yolov8n.pt", 
                        help="预训练模型路径 (默认: yolov8n.pt)")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="数据集配置文件 (默认: data.yaml)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小 (默认: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸 (默认: 640)")
    parser.add_argument("--patience", type=int, default=50,
                        help="早停耐心值 (默认: 50)")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="",
                        help="训练设备，如 0 或 0,1 或 cpu (默认: 自动选择)")
    parser.add_argument("--workers", type=int, default=8,
                        help="数据加载线程数 (默认: 8)")
    
    # 优化器参数
    parser.add_argument("--optimizer", type=str, default="auto",
                        choices=["SGD", "Adam", "AdamW", "auto"],
                        help="优化器 (默认: auto)")
    
    # 输出参数
    parser.add_argument("--name", type=str, default="elevator_ev",
                        help="实验名称 (默认: elevator_ev)")
    
    args = parser.parse_args()
    
    # 检查数据集是否存在
    if not os.path.exists(args.data):
        print(f"错误: 数据集配置文件 {args.data} 不存在")
        print("请先下载数据集并解压到 datasets/ 目录")
        print("下载链接: https://pan.baidu.com/s/1VJ-HAOjlYpfnz9UYA5R3vA?pwd=4mnw")
        print("提取码: 4mnw")
        exit(1)
    
    train(args)
