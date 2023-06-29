import cv2
import time


# 打开摄像头
cap = cv2.VideoCapture(0)
# 等待3秒钟
time.sleep(3)
# 读取一帧图像
ret, frame = cap.read()

# 如果成功读取到图像，则保存为文件
if ret:
    cv2.imwrite('image2.jpg', frame)

# 关闭摄像头
cap.release()
