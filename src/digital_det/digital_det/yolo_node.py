#!/usr/bin/env python3
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloPersonNode(Node):
    def __init__(self):
        super().__init__("yolo_person_node")

        self.declare_parameter("model_path", "/home/nedas/dev_ws/yolo26n_openvino_model")
        self.declare_parameter("conf", 0.4)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("max_det", 6)
        self.declare_parameter("person_class", 0)
        self.declare_parameter("input_topic", "/camera/intensity")
        self.declare_parameter("output_topic", "/det/persons")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("every_n", 2)
        self.declare_parameter("min_box_w", 20.0)
        self.declare_parameter("min_box_h", 20.0)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("half", False)
        self.declare_parameter("warmup_runs", 2)

        self.model_path   = str(self.get_parameter("model_path").value)
        self.conf         = float(self.get_parameter("conf").value)
        self.iou          = float(self.get_parameter("iou").value)
        self.max_det      = int(self.get_parameter("max_det").value)
        self.person_class = int(self.get_parameter("person_class").value)
        self.every_n      = int(self.get_parameter("every_n").value)
        self.imgsz        = int(self.get_parameter("imgsz").value)
        self.input_topic  = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.min_box_w    = float(self.get_parameter("min_box_w").value)
        self.min_box_h    = float(self.get_parameter("min_box_h").value)
        self.device       = str(self.get_parameter("device").value)
        self.use_half     = bool(self.get_parameter("half").value)
        self.warmup_runs  = int(self.get_parameter("warmup_runs").value)

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")

        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(max(1, self.warmup_runs)):
            self.model.predict(
                source=dummy,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                classes=[self.person_class],
                imgsz=self.imgsz,
                device=self.device,
                half=self.use_half,
                verbose=False
            )

        self.bridge = CvBridge()
        # sensor QoS keeps latency low and drops stale frames under load.
        self.sub = self.create_subscription(Image, self.input_topic, self.on_image, qos_profile_sensor_data)
        self.pub = self.create_publisher(Detection2DArray, self.output_topic, qos_profile_sensor_data)

        self.frame_i = 0        # received frames
        self.proc_i = 0         # frames processed by YOLO
        self.skip_i = 0         # dropped by every_n
        self.t0 = time.time()

        self.get_logger().info(
            f"Sub: {self.input_topic}  Pub: {self.output_topic}  imgsz={self.imgsz}  every_n={self.every_n} "
            f"device={self.device} half={self.use_half}"
        )

    def on_image(self, msg: Image):
        self.frame_i += 1
        if self.every_n > 1 and (self.frame_i % self.every_n) != 0:
            self.skip_i += 1
            return

        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge decode failed: {e}")
            return

        # OpenCV conversion is much faster than python-side channel stacking.
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        try:
            results = self.model.predict(
                source=bgr,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                classes=[self.person_class],
                imgsz=self.imgsz,
                device=self.device,
                half=self.use_half,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f"YOLO predict failed: {e}")
            return

        out = Detection2DArray()
        out.header = msg.header

        for r in results:
            if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)

                if bw < self.min_box_w or bh < self.min_box_h:
                    continue

                conf = float(box.conf[0]) if box.conf is not None else 0.0

                det = Detection2D()
                det.header = msg.header

                bb = BoundingBox2D()
                bb.center.position.x = float((x1 + x2) * 0.5)
                bb.center.position.y = float((y1 + y2) * 0.5)
                bb.size_x = float(bw)
                bb.size_y = float(bh)
                det.bbox = bb

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(self.person_class)
                hyp.hypothesis.score = conf
                det.results.append(hyp)

                out.detections.append(det)

        self.pub.publish(out)

        self.proc_i += 1
        dt = time.time() - self.t0
        if dt > 2.0:
            recv_fps = self.frame_i / dt
            yolo_fps = self.proc_i / dt
            self.get_logger().info(
                f"YOLO FPS ~ {yolo_fps:.1f}  input_fps ~ {recv_fps:.1f}  skipped={self.skip_i} "
                f"imgsz={self.imgsz} every_n={self.every_n} dets={len(out.detections)}"
            )
            self.t0 = time.time()
            self.frame_i = 0
            self.proc_i = 0
            self.skip_i = 0


def main():
    rclpy.init()
    node = YoloPersonNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
