#!/usr/bin/env python3
import time
import re
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32
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
        self.declare_parameter("input_topic", "/camera/intensity_rgb")
        self.declare_parameter("output_topic", "/det/persons")
        self.declare_parameter("imgsz", 512)
        self.declare_parameter("every_n", 1)
        self.declare_parameter("min_box_w", 20.0)
        self.declare_parameter("min_box_h", 20.0)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("half", False)
        self.declare_parameter("warmup_runs", 2)
        self.declare_parameter("qos_reliability", "best_effort")  # reliable | best_effort
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("latency_mode", False)
        self.declare_parameter("out_latency_topic", "/latency/yolo_ms")  # legacy total
        self.declare_parameter("out_infer_latency_topic", "/latency/yolo_infer_ms")
        self.declare_parameter("out_queue_latency_topic", "/latency/yolo_queue_ms")

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
        self.qos_reliability = str(self.get_parameter("qos_reliability").value).strip().lower()
        self.qos_depth = int(self.get_parameter("qos_depth").value)
        self.latency_mode = bool(self.get_parameter("latency_mode").value)
        self.out_latency_topic = str(self.get_parameter("out_latency_topic").value)
        self.out_infer_latency_topic = str(self.get_parameter("out_infer_latency_topic").value)
        self.out_queue_latency_topic = str(self.get_parameter("out_queue_latency_topic").value)
        self.qos = self._build_qos()

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")

        self._warmup_with_model_shape_fallback()

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.input_topic, self.on_image, self.qos)
        self.pub = self.create_publisher(Detection2DArray, self.output_topic, self.qos)
        self.pub_latency = self.create_publisher(Float32, self.out_latency_topic, self.qos) if self.latency_mode else None
        self.pub_latency_infer = (
            self.create_publisher(Float32, self.out_infer_latency_topic, self.qos) if self.latency_mode else None
        )
        self.pub_latency_queue = (
            self.create_publisher(Float32, self.out_queue_latency_topic, self.qos) if self.latency_mode else None
        )

        self.frame_i = 0        # received frames
        self.proc_i = 0         # frames processed by YOLO
        self.skip_i = 0         # dropped by every_n
        self.t0 = time.time()

        self.get_logger().info(
            f"Sub: {self.input_topic}  Pub: {self.output_topic}  imgsz={self.imgsz}  every_n={self.every_n} "
            f"device={self.device} half={self.use_half} qos={self.qos_reliability}"
        )
        if self.latency_mode:
            self.get_logger().info(
                f"Latency mode ON -> {self.out_latency_topic}, "
                f"{self.out_infer_latency_topic}, {self.out_queue_latency_topic}"
            )

    def _extract_model_square_size(self, err_text: str):
        m = re.search(r"model input \(shape=\[1,3,(\d+),(\d+)\]\)", err_text)
        if not m:
            return None
        h = int(m.group(1))
        w = int(m.group(2))
        if h == w and h > 0:
            return h
        return None

    def _warmup_once(self):
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
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

    def _warmup_with_model_shape_fallback(self):
        n = max(1, self.warmup_runs)
        for _ in range(n):
            try:
                self._warmup_once()
            except Exception as e:
                err = str(e)
                req = self._extract_model_square_size(err)
                if req is not None and req != self.imgsz:
                    self.get_logger().warn(
                        f"Model requires {req}x{req}, but imgsz={self.imgsz}. "
                        f"Switching imgsz to {req} automatically."
                    )
                    self.imgsz = req
                    self._warmup_once()
                else:
                    raise

    def _build_qos(self):
        reliability = ReliabilityPolicy.RELIABLE
        if self.qos_reliability == "best_effort":
            reliability = ReliabilityPolicy.BEST_EFFORT
        elif self.qos_reliability != "reliable":
            self.get_logger().warn(f"Unknown qos_reliability='{self.qos_reliability}', using 'reliable'")
            self.qos_reliability = "reliable"

        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, self.qos_depth),
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )

    def on_image(self, msg: Image):
        t0 = time.perf_counter()
        if self.pub_latency_queue is not None:
            now_ns = self.get_clock().now().nanoseconds
            msg_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            if msg_ns > 0 and now_ns >= msg_ns:
                q = Float32()
                q.data = float((now_ns - msg_ns) / 1e6)
                self.pub_latency_queue.publish(q)
        self.frame_i += 1
        if self.every_n > 1 and (self.frame_i % self.every_n) != 0:
            self.skip_i += 1
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"cv_bridge decode failed: {e}")
            return

        if img is None:
            return
        if len(img.shape) == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            if msg.encoding.lower() == "rgb8":
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img
        else:
            self.get_logger().warn(f"Unsupported image shape={img.shape}, encoding={msg.encoding}")
            return

        try:
            t_inf0 = time.perf_counter()
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
            infer_ms = float((time.perf_counter() - t_inf0) * 1000.0)
            if self.pub_latency_infer is not None:
                inf = Float32()
                inf.data = infer_ms
                self.pub_latency_infer.publish(inf)
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
        if self.pub_latency is not None:
            lat = Float32()
            lat.data = float((time.perf_counter() - t0) * 1000.0)
            self.pub_latency.publish(lat)

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
