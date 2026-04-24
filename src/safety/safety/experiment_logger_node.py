#!/usr/bin/env python3
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String, UInt8, Bool
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge


class ExperimentLoggerNode(Node):
    def __init__(self):
        super().__init__("experiment_logger_node")

        # input topics
        self.declare_parameter("z_topic", "/depth/person_z")
        self.declare_parameter("status_topic", "/depth/person_status")
        self.declare_parameter("detected_topic", "/depth/person_detected")
        self.declare_parameter("depth_valid_topic", "/depth/depth_valid")
        self.declare_parameter("in_zone_topic", "/depth/in_zone")
        self.declare_parameter("safety_state_topic", "/safety/state")
        self.declare_parameter("safety_level_topic", "/safety/level")
        self.declare_parameter("detections_topic", "/det/persons")
        self.declare_parameter("disparity_topic", "/camera/disparity")
        self.declare_parameter("detection_image_topic", "/camera/intensity_rgb")
        self.declare_parameter("scale_detection_to_disparity", True)
        self.declare_parameter("latency_mode", False)
        self.declare_parameter("camera_latency_topic", "/latency/camera_ms")  # legacy total
        self.declare_parameter("camera_intensity_latency_topic", "/latency/camera_intensity_ms")
        self.declare_parameter("camera_disparity_latency_topic", "/latency/camera_disparity_ms")
        self.declare_parameter("yolo_latency_topic", "/latency/yolo_ms")  # legacy total
        self.declare_parameter("yolo_infer_latency_topic", "/latency/yolo_infer_ms")
        self.declare_parameter("yolo_queue_latency_topic", "/latency/yolo_queue_ms")
        self.declare_parameter("depth_latency_topic", "/latency/depth_ms")  # legacy total
        self.declare_parameter("depth_crop_latency_topic", "/latency/depth_roi_crop_ms")
        self.declare_parameter("depth_distance_latency_topic", "/latency/depth_distance_ms")
        self.declare_parameter("depth_publish_latency_topic", "/latency/depth_publish_ms")
        self.declare_parameter("safety_latency_topic", "/latency/safety_ms")  # legacy total
        self.declare_parameter("safety_eval_latency_topic", "/latency/safety_eval_ms")
        self.declare_parameter("safety_publish_latency_topic", "/latency/safety_publish_ms")

        # experiment metadata
        self.declare_parameter("experiment_id", "exp")
        self.declare_parameter("lighting", "normal")
        self.declare_parameter("scenario", "default")
        self.declare_parameter("ground_truth_m", -1.0)
        self.declare_parameter("notes", "")

        # logging behavior
        self.declare_parameter("output_dir", "/home/nedas/dev_ws/experiment_logs")
        self.declare_parameter("session_name", "")
        self.declare_parameter("log_rate_hz", 10.0)
        self.declare_parameter("qos_reliability", "best_effort")  # reliable | best_effort
        self.declare_parameter("qos_depth", 20)
        self.declare_parameter("disp_scale", 16.0)
        self.declare_parameter("baseline_m", 0.160)
        self.declare_parameter("fx_px", 540.73152)
        self.declare_parameter("zmin", 0.50)
        self.declare_parameter("zmax", 3.00)
        self.declare_parameter("min_valid_depth_pixels", 6)

        self.z_topic = str(self.get_parameter("z_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.detected_topic = str(self.get_parameter("detected_topic").value)
        self.depth_valid_topic = str(self.get_parameter("depth_valid_topic").value)
        self.in_zone_topic = str(self.get_parameter("in_zone_topic").value)
        self.safety_state_topic = str(self.get_parameter("safety_state_topic").value)
        self.safety_level_topic = str(self.get_parameter("safety_level_topic").value)
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.disparity_topic = str(self.get_parameter("disparity_topic").value)
        self.detection_image_topic = str(self.get_parameter("detection_image_topic").value)
        self.scale_detection_to_disparity = bool(self.get_parameter("scale_detection_to_disparity").value)
        self.latency_mode = bool(self.get_parameter("latency_mode").value)
        self.camera_latency_topic = str(self.get_parameter("camera_latency_topic").value)
        self.camera_intensity_latency_topic = str(self.get_parameter("camera_intensity_latency_topic").value)
        self.camera_disparity_latency_topic = str(self.get_parameter("camera_disparity_latency_topic").value)
        self.yolo_latency_topic = str(self.get_parameter("yolo_latency_topic").value)
        self.yolo_infer_latency_topic = str(self.get_parameter("yolo_infer_latency_topic").value)
        self.yolo_queue_latency_topic = str(self.get_parameter("yolo_queue_latency_topic").value)
        self.depth_latency_topic = str(self.get_parameter("depth_latency_topic").value)
        self.depth_crop_latency_topic = str(self.get_parameter("depth_crop_latency_topic").value)
        self.depth_distance_latency_topic = str(self.get_parameter("depth_distance_latency_topic").value)
        self.depth_publish_latency_topic = str(self.get_parameter("depth_publish_latency_topic").value)
        self.safety_latency_topic = str(self.get_parameter("safety_latency_topic").value)
        self.safety_eval_latency_topic = str(self.get_parameter("safety_eval_latency_topic").value)
        self.safety_publish_latency_topic = str(self.get_parameter("safety_publish_latency_topic").value)

        self.experiment_id = str(self.get_parameter("experiment_id").value)
        self.lighting = str(self.get_parameter("lighting").value)
        self.scenario = str(self.get_parameter("scenario").value)
        self.ground_truth_m = float(self.get_parameter("ground_truth_m").value)
        self.notes = str(self.get_parameter("notes").value)

        self.output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        self.session_name = str(self.get_parameter("session_name").value).strip()
        self.log_rate_hz = float(self.get_parameter("log_rate_hz").value)
        self.qos_reliability = str(self.get_parameter("qos_reliability").value).strip().lower()
        self.qos_depth = int(self.get_parameter("qos_depth").value)
        self.disp_scale = float(self.get_parameter("disp_scale").value)
        self.baseline_m = float(self.get_parameter("baseline_m").value)
        self.fx_px = float(self.get_parameter("fx_px").value)
        self.zmin = float(self.get_parameter("zmin").value)
        self.zmax = float(self.get_parameter("zmax").value)
        self.min_valid_depth_pixels = int(self.get_parameter("min_valid_depth_pixels").value)

        self.qos = self._build_qos()
        self.bridge = CvBridge()
        self.latest_disp = None
        self.latest_det_img_w = None
        self.latest_det_img_h = None

        # latest values
        self.z = -1.0
        self.depth_status_text = ""
        self.person_detected = False
        self.depth_valid = False
        self.in_zone = False
        self.safety_state_text = ""
        self.safety_level = 0
        self.det_count = 0
        self.closest_conf = 0.0
        self.closest_cx = -1.0
        self.closest_cy = -1.0
        self.closest_w = -1.0
        self.closest_h = -1.0
        self.closest_depth_m = -1.0
        self.det_depth_valid_count = 0
        self.det_records = []
        self.state_changed = 0
        self._last_state = ""
        self.input_fps = 0.0
        self.output_fps = 0.0
        self._fps_window_start = time.monotonic()
        self._fps_input_count = 0
        self._fps_output_count = 0
        self._fps_det_count = 0
        self._fps_safety_state_count = 0
        self._last_status_print_t = 0.0
        self.det_fps = 0.0
        self.safety_state_hz = 0.0
        self.latest_disp_stamp_ns = None
        self.latest_det_stamp_ns = None
        self.e2e_latency_ms_disp = -1.0
        self.e2e_latency_ms_det = -1.0
        self.camera_node_ms = -1.0
        self.camera_intensity_ms = -1.0
        self.camera_disparity_ms = -1.0
        self.yolo_node_ms = -1.0
        self.yolo_infer_ms = -1.0
        self.yolo_queue_ms = -1.0
        self.depth_node_ms = -1.0
        self.depth_crop_ms = -1.0
        self.depth_distance_ms = -1.0
        self.depth_publish_ms = -1.0
        self.safety_node_ms = -1.0
        self.safety_eval_ms = -1.0
        self.safety_publish_ms = -1.0
        self.total_system_latency_ms = -1.0
        self.nodes_total_ms = -1.0

        self._init_csv()
        self._init_subscriptions()
        self._init_timer()

        self.get_logger().info(f"Experiment logger started -> {self.csv_path}")

    def _build_qos(self):
        reliability = ReliabilityPolicy.RELIABLE
        if self.qos_reliability == "best_effort":
            reliability = ReliabilityPolicy.BEST_EFFORT
        elif self.qos_reliability != "reliable":
            self.get_logger().warn(f"Unknown qos_reliability='{self.qos_reliability}', using 'best_effort'")
            reliability = ReliabilityPolicy.BEST_EFFORT

        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, self.qos_depth),
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )

    def _init_csv(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.session_name:
            self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        safe_exp = self.experiment_id.replace(" ", "_")
        file_name = f"{safe_exp}_{self.session_name}.csv"
        self.csv_path = self.output_dir / file_name

        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "wall_time_iso",
            "ros_time_s",
            "experiment_id",
            "lighting",
            "scenario",
            "ground_truth_m",
            "notes",
            "person_z_m",
            "person_detected",
            "depth_valid",
            "in_zone",
            "depth_status_text",
            "safety_state_text",
            "safety_level",
            "state_changed",
            "det_count",
            "det_depth_valid_count",
            "detections_json",
            "closest_det_conf",
            "closest_bbox_cx",
            "closest_bbox_cy",
            "closest_bbox_w",
            "closest_bbox_h",
            "closest_depth_m",
            "input_fps",
            "output_fps",
            "det_fps",
            "safety_state_hz",
            "e2e_latency_ms_disp_to_safety",
            "e2e_latency_ms_det_to_safety",
            "camera_node_ms",
            "camera_intensity_ms",
            "camera_disparity_ms",
            "yolo_node_ms",
            "yolo_infer_ms",
            "yolo_queue_ms",
            "depth_node_ms",
            "depth_crop_ms",
            "depth_distance_ms",
            "depth_publish_ms",
            "safety_node_ms",
            "safety_eval_ms",
            "safety_publish_ms",
            "total_system_latency_ms",
        ])
        self.csv_file.flush()

    def _init_subscriptions(self):
        self.create_subscription(Float32, self.z_topic, self.on_z, self.qos)
        self.create_subscription(String, self.status_topic, self.on_depth_status, self.qos)
        self.create_subscription(Bool, self.detected_topic, self.on_detected, self.qos)
        self.create_subscription(Bool, self.depth_valid_topic, self.on_depth_valid, self.qos)
        self.create_subscription(Bool, self.in_zone_topic, self.on_in_zone, self.qos)
        self.create_subscription(String, self.safety_state_topic, self.on_safety_state, self.qos)
        self.create_subscription(UInt8, self.safety_level_topic, self.on_safety_level, self.qos)
        self.create_subscription(Detection2DArray, self.detections_topic, self.on_detections, self.qos)
        self.create_subscription(Image, self.disparity_topic, self.on_disparity, self.qos)
        self.create_subscription(Image, self.detection_image_topic, self.on_detection_image, self.qos)
        if self.latency_mode:
            self.create_subscription(Float32, self.camera_latency_topic, self.on_camera_latency, self.qos)
            self.create_subscription(Float32, self.camera_intensity_latency_topic, self.on_camera_intensity_latency, self.qos)
            self.create_subscription(Float32, self.camera_disparity_latency_topic, self.on_camera_disparity_latency, self.qos)
            self.create_subscription(Float32, self.yolo_latency_topic, self.on_yolo_latency, self.qos)
            self.create_subscription(Float32, self.yolo_infer_latency_topic, self.on_yolo_infer_latency, self.qos)
            self.create_subscription(Float32, self.yolo_queue_latency_topic, self.on_yolo_queue_latency, self.qos)
            self.create_subscription(Float32, self.depth_latency_topic, self.on_depth_latency, self.qos)
            self.create_subscription(Float32, self.depth_crop_latency_topic, self.on_depth_crop_latency, self.qos)
            self.create_subscription(Float32, self.depth_distance_latency_topic, self.on_depth_distance_latency, self.qos)
            self.create_subscription(Float32, self.depth_publish_latency_topic, self.on_depth_publish_latency, self.qos)
            self.create_subscription(Float32, self.safety_latency_topic, self.on_safety_latency, self.qos)
            self.create_subscription(Float32, self.safety_eval_latency_topic, self.on_safety_eval_latency, self.qos)
            self.create_subscription(Float32, self.safety_publish_latency_topic, self.on_safety_publish_latency, self.qos)

    def _init_timer(self):
        hz = max(0.2, self.log_rate_hz)
        self.timer = self.create_timer(1.0 / hz, self.on_timer)

    def on_z(self, msg: Float32):
        self.z = float(msg.data)
        self._fps_input_count += 1

    def on_depth_status(self, msg: String):
        self.depth_status_text = str(msg.data)

    def on_detected(self, msg: Bool):
        self.person_detected = bool(msg.data)

    def on_depth_valid(self, msg: Bool):
        self.depth_valid = bool(msg.data)

    def on_in_zone(self, msg: Bool):
        self.in_zone = bool(msg.data)

    def on_safety_state(self, msg: String):
        new_state = str(msg.data)
        self.state_changed = 1 if self._last_state and new_state != self._last_state else 0
        self._last_state = new_state
        self.safety_state_text = new_state
        self._fps_safety_state_count += 1

        now_ns = self.get_clock().now().nanoseconds
        if self.latest_disp_stamp_ns is not None:
            dt_ms = (now_ns - self.latest_disp_stamp_ns) / 1e6
            if dt_ms >= 0.0:
                self.e2e_latency_ms_disp = float(dt_ms)
        if self.latest_det_stamp_ns is not None:
            dt_ms = (now_ns - self.latest_det_stamp_ns) / 1e6
            if dt_ms >= 0.0:
                self.e2e_latency_ms_det = float(dt_ms)

    def on_safety_level(self, msg: UInt8):
        self.safety_level = int(msg.data)

    def on_camera_latency(self, msg: Float32):
        self.camera_node_ms = float(msg.data)

    def on_camera_intensity_latency(self, msg: Float32):
        self.camera_intensity_ms = float(msg.data)

    def on_camera_disparity_latency(self, msg: Float32):
        self.camera_disparity_ms = float(msg.data)

    def on_yolo_latency(self, msg: Float32):
        self.yolo_node_ms = float(msg.data)

    def on_yolo_infer_latency(self, msg: Float32):
        self.yolo_infer_ms = float(msg.data)

    def on_yolo_queue_latency(self, msg: Float32):
        self.yolo_queue_ms = float(msg.data)

    def on_depth_latency(self, msg: Float32):
        self.depth_node_ms = float(msg.data)

    def on_depth_crop_latency(self, msg: Float32):
        self.depth_crop_ms = float(msg.data)

    def on_depth_distance_latency(self, msg: Float32):
        self.depth_distance_ms = float(msg.data)

    def on_depth_publish_latency(self, msg: Float32):
        self.depth_publish_ms = float(msg.data)

    def on_safety_latency(self, msg: Float32):
        self.safety_node_ms = float(msg.data)

    def on_safety_eval_latency(self, msg: Float32):
        self.safety_eval_ms = float(msg.data)

    def on_safety_publish_latency(self, msg: Float32):
        self.safety_publish_ms = float(msg.data)

    def _clamp_bbox(self, cx: float, cy: float, bw: float, bh: float, w: int, h: int):
        x1 = int(max(0, min(w - 1, round(cx - bw * 0.5))))
        y1 = int(max(0, min(h - 1, round(cy - bh * 0.5))))
        x2 = int(max(0, min(w, round(cx + bw * 0.5))))
        y2 = int(max(0, min(h, round(cy + bh * 0.5))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _scale_bbox_to_disparity(self, cx: float, cy: float, bw: float, bh: float, disp_w: int, disp_h: int):
        if not self.scale_detection_to_disparity:
            return cx, cy, bw, bh
        if self.latest_det_img_w is None or self.latest_det_img_h is None:
            return cx, cy, bw, bh
        src_w = float(self.latest_det_img_w)
        src_h = float(self.latest_det_img_h)
        if src_w <= 1.0 or src_h <= 1.0:
            return cx, cy, bw, bh

        sx = float(disp_w) / src_w
        sy = float(disp_h) / src_h
        return cx * sx, cy * sy, bw * sx, bh * sy

    def _estimate_depth_for_roi(self, roi):
        if self.latest_disp is None or roi is None:
            return -1.0

        x1, y1, x2, y2 = roi
        patch_u16 = self.latest_disp[y1:y2, x1:x2]
        if patch_u16.size <= 0:
            return -1.0

        d = patch_u16.astype(np.float32) / max(1e-6, self.disp_scale)
        d[d <= 0.0] = np.nan
        z = (self.fx_px * self.baseline_m) / d
        vals = z[np.isfinite(z)]
        vals = vals[(vals >= self.zmin) & (vals <= self.zmax)]
        if vals.size < max(1, self.min_valid_depth_pixels):
            return -1.0

        return float(np.median(vals))

    def on_disparity(self, msg: Image):
        try:
            disp = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return

        if disp is None or disp.dtype != np.uint16:
            return

        self.latest_disp = disp
        self.latest_disp_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def on_detection_image(self, msg: Image):
        self.latest_det_img_w = int(msg.width)
        self.latest_det_img_h = int(msg.height)

    def on_detections(self, msg: Detection2DArray):
        self._fps_det_count += 1
        self.latest_det_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.det_count = len(msg.detections)
        self.closest_conf = 0.0
        self.closest_cx = -1.0
        self.closest_cy = -1.0
        self.closest_w = -1.0
        self.closest_h = -1.0
        self.closest_depth_m = -1.0
        self.det_depth_valid_count = 0
        self.det_records = []

        if not msg.detections:
            return

        best = None
        best_area = -1.0
        disp_h = int(self.latest_disp.shape[0]) if self.latest_disp is not None else -1
        disp_w = int(self.latest_disp.shape[1]) if self.latest_disp is not None else -1
        for det in msg.detections:
            bw = float(det.bbox.size_x)
            bh = float(det.bbox.size_y)
            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            area = bw * bh
            if area > best_area:
                best_area = area
                best = det

            conf = float(det.results[0].hypothesis.score) if det.results else 0.0
            roi = None
            depth_m = -1.0
            depth_valid = False
            if disp_w > 0 and disp_h > 0:
                scx, scy, sbw, sbh = self._scale_bbox_to_disparity(cx, cy, bw, bh, disp_w, disp_h)
                roi = self._clamp_bbox(scx, scy, sbw, sbh, disp_w, disp_h)
                depth_m = self._estimate_depth_for_roi(roi)
                depth_valid = math.isfinite(depth_m) and depth_m > 0.0
                if depth_valid:
                    self.det_depth_valid_count += 1

            x1, y1, x2, y2 = (-1, -1, -1, -1) if roi is None else roi
            self.det_records.append({
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "w": round(bw, 2),
                "h": round(bh, 2),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": round(conf, 4),
                "depth_m": round(depth_m, 4) if depth_valid else None,
                "depth_valid": bool(depth_valid),
            })

        if best is None:
            return

        self.closest_cx = float(best.bbox.center.position.x)
        self.closest_cy = float(best.bbox.center.position.y)
        self.closest_w = float(best.bbox.size_x)
        self.closest_h = float(best.bbox.size_y)
        if disp_w > 0 and disp_h > 0:
            scx, scy, sbw, sbh = self._scale_bbox_to_disparity(
                self.closest_cx, self.closest_cy, self.closest_w, self.closest_h, disp_w, disp_h
            )
            closest_roi = self._clamp_bbox(scx, scy, sbw, sbh, disp_w, disp_h)
            self.closest_depth_m = self._estimate_depth_for_roi(closest_roi)
        if best.results:
            self.closest_conf = float(best.results[0].hypothesis.score)

    def on_timer(self):
        self._fps_output_count += 1
        now_mono = time.monotonic()
        elapsed = max(1e-6, now_mono - self._fps_window_start)
        self.input_fps = self._fps_input_count / elapsed
        self.output_fps = self._fps_output_count / elapsed
        self.det_fps = self._fps_det_count / elapsed
        self.safety_state_hz = self._fps_safety_state_count / elapsed

        if elapsed >= 1.0:
            self._fps_window_start = now_mono
            self._fps_input_count = 0
            self._fps_output_count = 0
            self._fps_det_count = 0
            self._fps_safety_state_count = 0

        now = self.get_clock().now()
        ros_time_s = now.nanoseconds / 1e9
        wall_time_iso = datetime.now().isoformat(timespec="milliseconds")
        node_parts = []
        for v in (self.camera_node_ms, self.yolo_node_ms, self.depth_node_ms, self.safety_node_ms):
            if v >= 0.0:
                node_parts.append(v)
        self.nodes_total_ms = float(sum(node_parts)) if node_parts else -1.0
        # Keep historical column name, but now reflect real sum of per-node processing ms.
        self.total_system_latency_ms = self.nodes_total_ms

        self.csv_writer.writerow([
            wall_time_iso,
            f"{ros_time_s:.6f}",
            self.experiment_id,
            self.lighting,
            self.scenario,
            f"{self.ground_truth_m:.3f}" if self.ground_truth_m >= 0.0 else "",
            self.notes,
            f"{self.z:.4f}",
            int(self.person_detected),
            int(self.depth_valid),
            int(self.in_zone),
            self.depth_status_text,
            self.safety_state_text,
            self.safety_level,
            self.state_changed,
            self.det_count,
            self.det_depth_valid_count,
            json.dumps(self.det_records, ensure_ascii=False, separators=(",", ":")),
            f"{self.closest_conf:.4f}",
            f"{self.closest_cx:.2f}",
            f"{self.closest_cy:.2f}",
            f"{self.closest_w:.2f}",
            f"{self.closest_h:.2f}",
            f"{self.closest_depth_m:.4f}" if self.closest_depth_m > 0.0 else "",
            f"{self.input_fps:.2f}",
            f"{self.output_fps:.2f}",
            f"{self.det_fps:.2f}",
            f"{self.safety_state_hz:.2f}",
            f"{self.e2e_latency_ms_disp:.2f}" if self.e2e_latency_ms_disp >= 0.0 else "",
            f"{self.e2e_latency_ms_det:.2f}" if self.e2e_latency_ms_det >= 0.0 else "",
            f"{self.camera_node_ms:.2f}" if self.camera_node_ms >= 0.0 else "",
            f"{self.camera_intensity_ms:.2f}" if self.camera_intensity_ms >= 0.0 else "",
            f"{self.camera_disparity_ms:.2f}" if self.camera_disparity_ms >= 0.0 else "",
            f"{self.yolo_node_ms:.2f}" if self.yolo_node_ms >= 0.0 else "",
            f"{self.yolo_infer_ms:.2f}" if self.yolo_infer_ms >= 0.0 else "",
            f"{self.yolo_queue_ms:.2f}" if self.yolo_queue_ms >= 0.0 else "",
            f"{self.depth_node_ms:.2f}" if self.depth_node_ms >= 0.0 else "",
            f"{self.depth_crop_ms:.2f}" if self.depth_crop_ms >= 0.0 else "",
            f"{self.depth_distance_ms:.2f}" if self.depth_distance_ms >= 0.0 else "",
            f"{self.depth_publish_ms:.2f}" if self.depth_publish_ms >= 0.0 else "",
            f"{self.safety_node_ms:.2f}" if self.safety_node_ms >= 0.0 else "",
            f"{self.safety_eval_ms:.2f}" if self.safety_eval_ms >= 0.0 else "",
            f"{self.safety_publish_ms:.2f}" if self.safety_publish_ms >= 0.0 else "",
            f"{self.total_system_latency_ms:.2f}" if self.total_system_latency_ms >= 0.0 else "",
        ])
        self.csv_file.flush()
        self.state_changed = 0

        if (now_mono - self._last_status_print_t) >= 1.0:
            z_text = f"{self.z:.2f} m" if self.depth_valid and self.z > 0.0 else "invalid"
            self.get_logger().info(
                f"time={wall_time_iso} | detected={int(self.person_detected)} | "
                f"depth_valid={int(self.depth_valid)} | distance={z_text} | "
                f"safety_state={self.safety_state_text or 'N/A'} | "
                f"bbox_cx={self.closest_cx:.2f} | conf={self.closest_conf:.3f} | "
                f"input_fps={self.input_fps:.2f} | output_fps={self.output_fps:.2f} | "
                f"safety_state_hz={self.safety_state_hz:.2f} | "
                f"cam_i_ms={self.camera_intensity_ms:.1f} cam_d_ms={self.camera_disparity_ms:.1f} "
                f"yolo_ms={self.yolo_node_ms:.1f} yolo_infer_ms={self.yolo_infer_ms:.1f} "
                f"depth_crop_ms={self.depth_crop_ms:.1f} depth_dist_ms={self.depth_distance_ms:.1f} "
                f"depth_pub_ms={self.depth_publish_ms:.1f} safety_eval_ms={self.safety_eval_ms:.1f} "
                f"safety_pub_ms={self.safety_publish_ms:.1f} | "
                f"nodes_total_ms={self.nodes_total_ms:.1f}"
            )
            self._last_status_print_t = now_mono

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ExperimentLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
