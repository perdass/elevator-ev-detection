"""
电梯内电动车检测 - PyQt5 可视化界面
"""
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QMessageBox, QStatusBar)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO
import os

class DetectionThread(QThread):
    """检测线程"""
    frame_ready = pyqtSignal(np.ndarray, list)
    
    def __init__(self, model_path, conf=0.5, iou=0.45):
        super().__init__()
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.running = False
        self.source = 0
        
    def set_source(self, source):
        self.source = source
        
    def update_params(self, conf, iou):
        self.conf = conf
        self.iou = iou
        
    def run(self):
        self.running = True
        
        if isinstance(self.source, str) and os.path.isfile(self.source):
            # 图片检测
            results = self.model(self.source, conf=self.conf, iou=self.iou)
            if results:
                img = results[0].orig_img.copy()
                detections = []
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((cls, conf, (x1, y1, x2, y2)))
                self.frame_ready.emit(img, detections)
        else:
            # 视频/摄像头检测
            cap = cv2.VideoCapture(self.source)
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
                detections = []
                
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((cls, conf, (x1, y1, x2, y2)))
                
                self.frame_ready.emit(frame, detections)
                self.msleep(30)  # ~30 FPS
            
            cap.release()
    
    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("电梯电动车检测系统")
        self.setMinimumSize(1000, 700)
        
        # 类别配置
        self.class_names = {0: "自行车", 1: "电动车"}
        self.class_colors = {
            0: (0, 255, 0),    # 绿色
            1: (0, 0, 255),    # 红色
        }
        
        # 初始化检测线程
        self.detector = None
        self.current_frame = None
        
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        central = QWidget()
        self.setCentralWidget(central)
        
        # 主布局
        main_layout = QHBoxLayout(central)
        
        # 左侧 - 视频显示
        video_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.video_label.setText("请选择输入源")
        video_layout.addWidget(self.video_label)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.btn_camera = QPushButton("打开摄像头")
        self.btn_camera.clicked.connect(self.open_camera)
        btn_layout.addWidget(self.btn_camera)
        
        self.btn_video = QPushButton("打开视频")
        self.btn_video.clicked.connect(self.open_video)
        btn_layout.addWidget(self.btn_video)
        
        self.btn_image = QPushButton("打开图片")
        self.btn_image.clicked.connect(self.open_image)
        btn_layout.addWidget(self.btn_image)
        
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)
        
        video_layout.addLayout(btn_layout)
        main_layout.addLayout(video_layout, 7)
        
        # 右侧 - 参数设置
        right_panel = QVBoxLayout()
        
        # 模型选择
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        
        model_path_layout = QHBoxLayout()
        self.model_label = QLabel("模型: yolov8n.pt")
        model_path_layout.addWidget(self.model_label)
        self.btn_model = QPushButton("选择模型")
        self.btn_model.clicked.connect(self.select_model)
        model_path_layout.addWidget(self.btn_model)
        model_layout.addLayout(model_path_layout)
        
        self.model_path = "yolov8n.pt"
        model_group.setLayout(model_layout)
        right_panel.addWidget(model_group)
        
        # 检测参数
        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout()
        
        # 置信度
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setSingleStep(0.05)
        conf_layout.addWidget(self.conf_spin)
        param_layout.addLayout(conf_layout)
        
        # IOU
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IOU:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.05)
        iou_layout.addWidget(self.iou_spin)
        param_layout.addLayout(iou_layout)
        
        # 电动车警报阈值
        alert_layout = QHBoxLayout()
        alert_layout.addWidget(QLabel("警报阈值:"))
        self.alert_spin = QDoubleSpinBox()
        self.alert_spin.setRange(0.1, 1.0)
        self.alert_spin.setValue(0.7)
        self.alert_spin.setSingleStep(0.05)
        alert_layout.addWidget(self.alert_spin)
        param_layout.addLayout(alert_layout)
        
        param_group.setLayout(param_layout)
        right_panel.addWidget(param_group)
        
        # 检测结果统计
        stats_group = QGroupBox("检测统计")
        stats_layout = QVBoxLayout()
        
        self.bicycle_count = QLabel("自行车: 0")
        self.bicycle_count.setFont(QFont("Arial", 12))
        stats_layout.addWidget(self.bicycle_count)
        
        self.ebike_count = QLabel("电动车: 0")
        self.ebike_count.setFont(QFont("Arial", 12))
        self.ebike_count.setStyleSheet("color: red;")
        stats_layout.addWidget(self.ebike_count)
        
        self.alert_label = QLabel("")
        self.alert_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.alert_label.setStyleSheet("color: red;")
        stats_layout.addWidget(self.alert_label)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        # 截图按钮
        self.btn_snapshot = QPushButton("保存截图")
        self.btn_snapshot.clicked.connect(self.save_snapshot)
        self.btn_snapshot.setEnabled(False)
        right_panel.addWidget(self.btn_snapshot)
        
        right_panel.addStretch()
        main_layout.addLayout(right_panel, 3)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def select_model(self):
        """选择模型文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "", "模型文件 (*.pt *.onnx);;所有文件 (*)"
        )
        if path:
            self.model_path = path
            self.model_label.setText(f"模型: {os.path.basename(path)}")
            self.statusBar().showMessage(f"已加载模型: {path}")
    
    def open_camera(self):
        """打开摄像头"""
        self.start_detection(0)
        
    def open_video(self):
        """打开视频文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if path:
            self.start_detection(path)
    
    def open_image(self):
        """打开图片"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if path:
            self.start_detection(path)
    
    def start_detection(self, source):
        """开始检测"""
        self.stop_detection()
        
        try:
            self.detector = DetectionThread(
                self.model_path,
                conf=self.conf_spin.value(),
                iou=self.iou_spin.value()
            )
            self.detector.frame_ready.connect(self.update_frame)
            self.detector.set_source(source)
            self.detector.start()
            
            self.btn_stop.setEnabled(True)
            self.btn_snapshot.setEnabled(True)
            self.statusBar().showMessage("检测中...")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
    
    def stop_detection(self):
        """停止检测"""
        if self.detector:
            self.detector.stop()
            self.detector = None
        
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("已停止")
    
    def update_frame(self, frame, detections):
        """更新画面"""
        self.current_frame = frame.copy()
        
        # 绘制检测结果
        bicycle_num = 0
        ebike_num = 0
        
        for cls, conf, (x1, y1, x2, y2) in detections:
            color = self.class_colors.get(cls, (255, 255, 255))
            
            # 绘制边框
            if cls == 1 and conf > self.alert_spin.value():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # 警告文字
                cv2.putText(frame, "WARNING!", (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            label = f"{self.class_names.get(cls, '?')} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if cls == 0:
                bicycle_num += 1
            elif cls == 1:
                ebike_num += 1
        
        # 更新统计
        self.bicycle_count.setText(f"自行车: {bicycle_num}")
        self.ebike_count.setText(f"电动车: {ebike_num}")
        
        # 警报
        if ebike_num > 0:
            self.alert_label.setText("⚠️ 检测到电动车!")
        else:
            self.alert_label.setText("")
        
        # 转换并显示
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放适应窗口
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)
    
    def save_snapshot(self):
        """保存截图"""
        if self.current_frame is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "保存截图", "snapshot.jpg", "图片 (*.jpg *.png)"
        )
        if path:
            cv2.imwrite(path, self.current_frame)
            self.statusBar().showMessage(f"截图已保存: {path}")
    
    def closeEvent(self, event):
        """关闭窗口"""
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QPushButton {
            padding: 8px 16px;
            font-size: 14px;
            border-radius: 4px;
            background-color: #4a90d9;
            color: white;
        }
        QPushButton:hover {
            background-color: #357abd;
        }
        QPushButton:disabled {
            background-color: #cccccc;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
