import numpy as np
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

import time

WIDTH = 480
HEIGHT = 320
FRAME_RATE = 60
ROT_DEG = 90

GAUSSIAN_SIZE = (15, 15)

MOV_ALPHA = 0.8
THRESH = 15
MIN_AREA = 600

CAP_INTERVAL = 5


def detect_moving_body(frame, avg):
    """
    動体検知
    """
    detected = False

    # 過去フレームと現在フレームの加重平均をとり、差分計算
    cv2.accumulateWeighted(gray, avg, MOV_ALPHA)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # 閾値処理で2値化
    thresh = cv2.threshold(frameDelta, THRESH, 255, cv2.THRESH_BINARY)[1]

    # 膨張処理後、閾値範囲の面積を取得
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts[1]:
        # 閾値範囲(動体部分)が最小面積以上なら検知
        if cv2.contourArea(c) < MIN_AREA:
            continue
        detected = True
        break

    return detected


if __name__ == '__main__':
    # picameraライブラリ
    with PiCamera() as camera:
        with PiRGBArray(camera, size=(HEIGHT, WIDTH)) as rawCapture:
            time.sleep(1)

            # 画像回転に合わせて縦、横の解像度調整
            if (ROT_DEG // 90) % 2 == 0:
                camera.resolution = (WIDTH, HEIGHT)
            elif (ROT_DEG // 90) % 2 == 1:
                camera.resolution = (HEIGHT, WIDTH)
            # フレームレート
            camera.framerate = FRAME_RATE
            time.sleep(1)

            avg = None
            # OpenCVに合わせてFormatをbgr
            for stream in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

                # Numpy形式で画像回転
                frame = stream.array
                frame = np.rot90(frame, k=ROT_DEG // 90)

                # グレースケールにし、フィルター処理
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, GAUSSIAN_SIZE, 1)

                if avg is None:
                    avg = gray.copy().astype("float")
                    rawCapture.truncate(0)
                    continue

                # 動体検知
                detected = detect_moving_body(frame, avg)
                if detected:
                    text = 'detecion'
                else:
                    text = 'not detection'

                frame = frame.astype(np.uint8).copy()
                cv2.putText(frame, "Room Status: {}".format(text), (10, 10),
                            cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
                cv2.imshow('video', frame)

                rawCapture.truncate(0)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # インターバル
                time.sleep(CAP_INTERVAL)

    cv2.destroyAllWindows()
