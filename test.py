import roslibpy
import numpy as np
import cv2
import base64
import matplotlib.pyplot as plt

client = roslibpy.Ros(host='192.168.0.244', port=9091)
client.run()

def cb(msg):
    b = base64.b64decode(msg["data"])          # ✅ 진짜 이미지 bytes로 복원
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plt.imshow(img) 
    plt.show()


def cb_depth(msg):
    # msg['data'] 는 base64 문자열인 경우가 일반적
    raw = base64.b64decode(msg['data'])

    h = msg['height']
    w = msg['width']
    enc = msg.get('encoding', '')

    if enc not in ('32FC1', '32FC1'):
        print('encoding:', enc)

    depth = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
    print(depth.shape, depth.dtype, np.nanmin(depth), np.nanmax(depth))

# topic = roslibpy.Topic(client, '/right/camera/cam_top/color/image_rect_raw/compressed', 'sensor_msgs/msg/CompressedImage')
# topic.subscribe(cb)

topic2 = roslibpy.Topic(client, '/right/camera/cam_top/depth/image_rect_raw', 'sensor_msgs/msg/Image')
topic2.subscribe(cb_depth)
while True:
    pass
