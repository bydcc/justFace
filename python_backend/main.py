import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from camera import video_camera, Model
from sio import sio_app, sio_server
import cv2
import asyncio
from core import TT

t = None


@sio_server.event
async def connect(sid, env, auth):
    print(f'{sid}: connected')


@sio_server.event
async def disconnect(sid):
    print(f'{sid}: disconnected')


@sio_server.on("camera.release")
async def releaseCamera(sid):
    # 在应用程序关闭时，关闭摄像头等资源
    video_camera.release_camera()


async def long_running_process():
    global t
    if not t:
        t = TT()
    # 处理时间很长的接口
    for i in range(100):
        await asyncio.sleep(1)
        await sio_server.emit('progress.update', i)

    print("long_running_process finished")


@sio_server.on('process.start')
async def handle_start(sid, data):
    print('data', data)
    task = asyncio.create_task(long_running_process())
    return {"message": "long_running_process started"}


async def interrupt_process():
    # 中断前面接口的协程
    tasks = asyncio.all_tasks()
    for task in tasks:
        if task != asyncio.current_task():
            task.cancel()


@sio_server.on('process.cancel')
async def handle_start(sid):
    asyncio.create_task(interrupt_process())
    return {"message": "interruption requested"}


app = FastAPI()
app.mount("/sio", app=sio_app)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/swapped_camera")
async def swapped_camera():
    return StreamingResponse(
        Model(video_camera.gen_frame()),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
