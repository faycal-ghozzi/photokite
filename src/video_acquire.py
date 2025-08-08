import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import array
import sys
import os
import threading
import queue

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config_node import *

class VideoAcquireNode(Node):
    """
    Noeud ROS2 pour acquisition RTSP, publication ROS et affichage OpenCV en thread séparé.
    """

    def __init__(self, params):
        super().__init__(params["NODE_NAME"] + "_video_acquire")

        self.VIDEO_URL = params["VIDEO_URL"]
        self.topicOutName = params["topicOutName"]
        self.comppressedMaxTime = params["comppressedMaxTime"]
        self.compressionRatio = int(params["compressionRatio"])
        self.mtx = params["mtx"]
        self.dist = params["dist"]
        self.HFOVInDeg = params["HFOVInDeg"]
        self.VFOVInDeg = params["VFOVInDeg"]
        self.frequency = params["frequency"]
        self.bridge = CvBridge()
        self.last_compress_time = time.time()
        self.cap = None
        self.is_first = True
        self.last_attempt_time = 0
        self.retry_delay = 5

        self.pub_img = self.create_publisher(Image, f"{params['NODE_NAME']}/{self.topicOutName}", 10)
        self.pub_compressed = self.create_publisher(CompressedImage, f"{params['NODE_NAME']}/{self.topicOutName}_compressed", 10)
        self.pub_info = self.create_publisher(CameraInfo, f"{params['NODE_NAME']}/{self.topicOutName}_info", 10)

        self.frame_queue = queue.Queue(maxsize=1)
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()

        timer_period = self.frequency if self.frequency > 0 else 0.05
        self.timer = self.create_timer(timer_period, self.process_frame)
        self.get_logger().info(f"[{self.get_name()}] Démarré avec source: {self.VIDEO_URL}")

    def calculate_mtx(self, width, height):
        fx = (width / 2.0) / np.tan((np.pi * self.HFOVInDeg / 180.0) / 2.0)
        fy = (height / 2.0) / np.tan((np.pi * self.VFOVInDeg / 180.0) / 2.0)
        cx = width / 2.0
        cy = height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def connect_stream(self):
        self.cap = cv2.VideoCapture(self.VIDEO_URL, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.get_logger().info("Flux vidéo connecté avec succès.")
        else:
            self.get_logger().error("Échec de connexion au flux vidéo.")
            self.cap = None

    def process_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().info("Connexion au flux...")
            self.cap = cv2.VideoCapture(self.VIDEO_URL, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                self.get_logger().info("Flux vidéo connecté.")
            else:
                self.get_logger().error("Impossible d’ouvrir le flux.")
                self.cap = None
                return

        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            self.get_logger().warning("⚠️ Flux interrompu, tentative de reconnexion immédiate.")
            self.cap.release()
            self.cap = None
            return

        height, width = frame.shape[:2]

        if self.is_first:
            cam_info = CameraInfo()
            cam_info.header.frame_id = self.get_name()
            cam_info.height = height
            cam_info.width = width

            if self.mtx is None:
                self.mtx = self.calculate_mtx(width, height)
            cam_info.k = array.array('d', self.mtx.flatten())

            if self.dist is not None:
                cam_info.d = array.array('d', self.dist.flatten())

            self.pub_info.publish(cam_info)
            self.is_first = False

        msg_img = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        msg_img.header.frame_id = self.get_name()
        self.pub_img.publish(msg_img)

        if self.comppressedMaxTime > 0 and (time.time() - self.last_compress_time) >= self.comppressedMaxTime:
            success, encimg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.compressionRatio])
            if success:
                msg_cmp = CompressedImage()
                msg_cmp.header.frame_id = self.get_name()
                msg_cmp.format = "jpeg"
                msg_cmp.data = encimg.tobytes()
                self.pub_compressed.publish(msg_cmp)
                self.last_compress_time = time.time()

        try:
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame.copy())
        except Exception as e:
            self.get_logger().warning(f"Erreur file affichage : {e}")


    def display_loop(self):
        cv2.namedWindow("Vue Caméra RTSP", cv2.WINDOW_NORMAL)
        while True:
            try:
                frame = self.frame_queue.get(timeout=1)
                cv2.imshow("Vue Caméra RTSP", frame)
                cv2.waitKey(1)
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().warning(f"Erreur dans display_loop: {e}")
                break


def main(args=None):
    rclpy.init(args=args)
    NODE_ID = eval(sys.argv[1])
    node = VideoAcquireNode(LIST_OF_COMPONENT[NODE_ID])

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
