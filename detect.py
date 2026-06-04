"""
电梯内电动车检测 - 推理脚本
"""
from ultralytics import YOLO
import argparse
import cv2
import os
from pathlib import Path

def detect(args):
    # 加载模型
    model = YOLO(args.model)
    
    # 类别名称
    class_names = {0: "bicycle", 1: "ebike"}
    class_names_cn = {0: "自行车", 1: "电动车"}
    
    # 颜色配置 (BGR)
    colors = {
        0: (0, 255, 0),    # 自行车 - 绿色
        1: (0, 0, 255),    # 电动车 - 红色
    }
    
    # 处理输入源
    source = args.source
    
    # 判断输入类型
    if source.isdigit():
        # 摄像头
        cap = cv2.VideoCapture(int(source))
    elif source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # 视频文件
        cap = cv2.VideoCapture(source)
    elif os.path.isdir(source):
        # 图片目录
        cap = None
    elif os.path.isfile(source):
        # 单张图片
        cap = None
    else:
        print(f"错误: 无法识别的输入源 {source}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if cap is None:
        # 图片检测
        results = model(source, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
        
        for i, result in enumerate(results):
            # 绘制检测结果
            img = result.orig_img.copy()
            
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 绘制边框
                color = colors.get(cls, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{class_names_cn.get(cls, 'unknown')} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存结果
            if os.path.isfile(source):
                output_path = output_dir / f"result_{Path(source).name}"
            else:
                output_path = output_dir / f"result_{i}.jpg"
            
            cv2.imwrite(str(output_path), img)
            print(f"检测结果保存到: {output_path}")
    
    else:
        # 视频/摄像头检测
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建视频写入器
        if args.save_video:
            output_video = output_dir / "result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测
            results = model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
            
            # 绘制结果
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 检查是否是电动车，触发警报
                if cls == 1 and conf > args.alert_conf:
                    color = (0, 0, 255)  # 红色警告
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # 添加警告文字
                    cv2.putText(frame, "WARNING: EBike Detected!", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    color = colors.get(cls, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{class_names_cn.get(cls, 'unknown')} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 显示画面
            if args.show:
                cv2.imshow("Elevator EV Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 保存视频
            if args.save_video:
                writer.write(frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count} 帧")
        
        cap.release()
        if args.save_video:
            writer.release()
            print(f"结果视频保存到: {output_video}")
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="电梯电动车检测推理")
    
    # 输入参数
    parser.add_argument("--source", type=str, default="0",
                        help="输入源 (图片/视频/摄像头) (默认: 0)")
    parser.add_argument("--model", type=str, default="runs/detect/elevator_ev/weights/best.pt",
                        help="模型路径 (默认: runs/detect/elevator_ev/weights/best.pt)")
    
    # 检测参数
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值 (默认: 0.5)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IOU 阈值 (默认: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸 (默认: 640)")
    parser.add_argument("--alert-conf", type=float, default=0.7,
                        help="电动车警报置信度阈值 (默认: 0.7)")
    
    # 输出参数
    parser.add_argument("--output", type=str, default="results",
                        help="输出目录 (默认: results)")
    parser.add_argument("--show", action="store_true",
                        help="显示检测结果")
    parser.add_argument("--save-video", action="store_true",
                        help="保存结果视频")
    
    args = parser.parse_args()
    
    detect(args)
