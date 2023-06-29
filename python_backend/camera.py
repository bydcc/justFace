import cv2


class VideoCamera:
    def __init__(self):
        self.video = None
    def get_camera(self):
        if not self.video:
            self.video = cv2.VideoCapture(0)

    def release_camera(self):
        if self.video:
            self.video.release()
        self.video = None

    def get_frame(self):
        _, frame = self.video.read()
        # 三维矩阵
        return frame

    def gen_frame(self):
        self.get_camera()
        while True:
            _, frame = self.video.read()
            # 三维矩阵
            yield frame

    def get_frame_bytes(self):
        self.get_camera()
        while True:
            _, frame = self.video.read()
            yield frame2bytes(frame)

video_camera = VideoCamera()

def frame2bytes(frame):
    _, jpeg = cv2.imencode(".jpg", frame)
    jpeg_by = jpeg.tobytes()
    return (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg_by + b"\r\n")



class Model:
    def __init__(self, generator):
        self.generator = generator
    
    def __iter__(self):
        return self
    
    def __next__(self):
        data = next(self.generator)
        return process(data)
    
def process(frame):
    frame = cv2.flip(frame, 1)
    return frame2bytes(frame)