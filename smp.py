import sys
import socket
import numpy as np
import pyqtgraph as pg
# 我们使用 PyQt5 作为 GUI 框架
from PyQt5 import QtCore, QtWidgets

# --- 网络配置 ---
# !!! 把它改成你树莓派的 IP 地址 !!!
PI_ADDRESS = '192.168.137.2' 
PORT = 9999

# --- 音频配置 (必须与服务器匹配) ---
CHUNK_FRAMES = 1024          # 每次读取的帧数
CHANNELS = 6                 # 通道数
BYTES_PER_SAMPLE = 4         # 4 字节 (paInt32)
RATE = 48000                 # 采样率

# 网络数据包的总大小
CHUNK_SIZE = CHUNK_FRAMES * CHANNELS * BYTES_PER_SAMPLE

# ---------------------------------------------------------------------
#   网络工作线程
#   (我们必须在单独的“线程”中接收网络数据，否则 GUI 窗口会卡死)
# ---------------------------------------------------------------------
class NetworkWorker(QtCore.QObject):
    # 'pyqtSignal' 是一种安全的方式，用于从该线程发送数据到主 GUI 线程
    # 我们将用它来发送 numpy 数组
    data_ready = QtCore.pyqtSignal(np.ndarray)
    
    # 发送状态信息（如 "Connected"）
    connection_status = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def run(self):
        """这个函数将在后台线程中运行"""
        try:
            self.connection_status.emit(f"正在连接 {PI_ADDRESS}:{PORT}...")
            self.sock.connect((PI_ADDRESS, PORT))
            self.connection_status.emit("已连接！正在接收音频流...")
            
            while self.running:
                # 1. 从网络接收原始字节
                data_bytes = self.sock.recv(CHUNK_SIZE)
                if not data_bytes:
                    break # 连接已关闭
                
                # 2. 将字节转换为 numpy 整数数组
                data_int = np.frombuffer(data_bytes, dtype=np.int32)
                
                # 3. 发送信号，把数据交给 GUI 线程
                self.data_ready.emit(data_int)

        except Exception as e:
            self.connection_status.emit(f"错误: {e}")
        finally:
            self.sock.close()
            self.connection_status.emit("连接已关闭。")

    def stop(self):
        self.running = False

# ---------------------------------------------------------------------
#   主 GUI 窗口
# ---------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- 1. 设置 GUI 窗口 ---
        self.setWindowTitle("实时音频波形图 (来自树莓派)")
        self.win = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.win)
        self.resize(1000, 400)
        
        # --- 2. 创建绘图区域 ---
        self.plot = self.win.addPlot(title="实时波形")
        # 设置Y轴范围为32位整数的全范围
        self.plot.setYRange(-(2**31), 2**31, padding=0.05) 
        self.plot.setXRange(0, CHUNK_FRAMES, padding=0.01)
        self.plot.setLabel('left', '幅值 (Amplitude)')
        self.plot.setLabel('bottom', '采样点 (Samples)')
        
        # --- 3. 创建那条“线” ---
        # 我们只绘制一个通道（左通道）的数据
        self.curve = self.plot.plot(pen='y') # 'y' = 黄色
        
        # 创建 X 轴数据 (这个是固定的，0 到 1023)
        self.x_data = np.arange(CHUNK_FRAMES)
        
        # --- 4. 添加状态栏 ---
        self.statusBar = self.statusBar()
        
        # --- 5. 设置网络线程 ---
        self.thread = QtCore.QThread()
        self.worker = NetworkWorker()
        self.worker.moveToThread(self.thread) # 将 worker 移动到新线程

        # --- 6. 连接信号和槽 ---
        self.thread.started.connect(self.worker.run) # 线程启动时，运行 worker.run
        self.worker.data_ready.connect(self.update_plot) # 当 worker 拿到数据时，调用 self.update_plot
        self.worker.connection_status.connect(self.set_status) # 当 worker 有状态更新时，调用 self.set_status

        # 启动线程
        self.thread.start()

    def update_plot(self, data_int):
        """这个函数会在主 GUI 线程中被调用"""
        # data_int 是交错的 [左, 右, 左, 右, ...]
        # 我们只取左通道：从索引0开始，每隔2个取一个
        left_channel_data = data_int[::CHANNELS]
        
        # 确保数据长度正确
        if len(left_channel_data) == len(self.x_data):
            # 更新曲线数据！
            self.curve.setData(self.x_data, left_channel_data)
            
    def set_status(self, message):
        self.statusBar.showMessage(message)

    def closeEvent(self, event):
        """在关闭窗口时被调用，用于安全地停止后台线程"""
        self.worker.stop()
        self.thread.quit()
        self.thread.wait() # 等待线程完全退出
        event.accept()

# --- 启动应用程序 ---
app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())