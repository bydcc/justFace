from sio import sio
from face_extract import FaceExtractor, VideoFrameExtractor


@sio.on('msg')
def handle_msg(sid, msg):
    print('received msg: ' + msg)
    sio.emit('msg', msg)


@sio.on('extractFace')
def handle_msg(sid, msg):
    sio.emit('extractFace', {'msg': '开始提取图片'})
    vide_frame_extractor = VideoFrameExtractor(
        msg['video_folder'], msg['output_folder'])
    vide_frame_extractor.extract_frames()
    sio.emit('extractFace', {'msg': '开始识别人脸'})
