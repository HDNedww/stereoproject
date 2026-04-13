#!/usr/bin/env python3
import csv
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32, String, UInt8, Bool
from vision_msgs.msg import Detection2DArray


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

        self.z_topic = str(self.get_parameter("z_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.detected_topic = str(self.get_parameter("detected_topic").value)
        self.depth_valid_topic = str(self.get_parameter("depth_valid_topic").value)
        self.in_zone_topic = str(self.get_parameter("in_zone_topic").value)
        self.safety_state_topic = str(self.get_parameter("safety_state_topic").value)
        self.safety_level_topic = str(self.get_parameter("safety_level_topic").value)
        self.detections_topic = str(self.get_parameter("detections_topic").value)

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

        self.qos = self._build_qos()

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
        self.state_changed = 0
        self._last_state = ""

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
            "closest_det_conf",
            "closest_bbox_cx",
            "closest_bbox_cy",
            "closest_bbox_w",
            "closest_bbox_h",
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

    def _init_timer(self):
        hz = max(0.2, self.log_rate_hz)
        self.timer = self.create_timer(1.0 / hz, self.on_timer)

    def on_z(self, msg: Float32):
        self.z = float(msg.data)

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

    def on_safety_level(self, msg: UInt8):
        self.safety_level = int(msg.data)

    def on_detections(self, msg: Detection2DArray):
        self.det_count = len(msg.detections)
        self.closest_conf = 0.0
        self.closest_cx = -1.0
        self.closest_cy = -1.0
        self.closest_w = -1.0
        self.closest_h = -1.0

        if not msg.detections:
            return

        best = None
        best_area = -1.0
        for det in msg.detections:
            bw = float(det.bbox.size_x)
            bh = float(det.bbox.size_y)
            area = bw * bh
            if area > best_area:
                best_area = area
                best = det

        if best is None:
            return

        self.closest_cx = float(best.bbox.center.position.x)
        self.closest_cy = float(best.bbox.center.position.y)
        self.closest_w = float(best.bbox.size_x)
        self.closest_h = float(best.bbox.size_y)
        if best.results:
            self.closest_conf = float(best.results[0].hypothesis.score)

    def on_timer(self):
        now = self.get_clock().now()
        ros_time_s = now.nanoseconds / 1e9
        wall_time_iso = datetime.now().isoformat(timespec="milliseconds")

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
            f"{self.closest_conf:.4f}",
            f"{self.closest_cx:.2f}",
            f"{self.closest_cy:.2f}",
            f"{self.closest_w:.2f}",
            f"{self.closest_h:.2f}",
        ])
        self.csv_file.flush()
        self.state_changed = 0

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
