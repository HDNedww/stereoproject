#!/usr/bin/env python3
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String, Bool
from vision_msgs.msg import Detection2DArray

from cv_bridge import CvBridge


class DepthNode(Node):
    def __init__(self):
        super().__init__("depth_node")

        # ---- stereo / depth params ----
        self.declare_parameter("disp_scale", 16.0)
        self.declare_parameter("baseline_m", 0.0650542)
        self.declare_parameter("fx_px", 540.73152)
        self.declare_parameter("fx_reference_width_px", 0.0)
        self.declare_parameter("zmin", 0.50)
        self.declare_parameter("zmax", 3.00)

        # ---- processing params ----
        self.declare_parameter("show_debug", True)
        self.declare_parameter("min_valid_depth_pixels", 6)
        self.declare_parameter("depth_percentile", 10.0)
        self.declare_parameter("depth_stat", "median")  # median | percentile
        self.declare_parameter("mad_gate_sigma", 2.5)   # 0 disables outlier rejection
        self.declare_parameter("max_det_age_ms", 600)
        self.declare_parameter("smooth_depth", True)
        self.declare_parameter("z_alpha", 0.7)
        self.declare_parameter("roi_mode", "full")  # full | torso
        self.declare_parameter("torso_side_frac", 0.20)
        self.declare_parameter("torso_top_frac", 0.15)
        self.declare_parameter("torso_bottom_frac", 0.88)
        self.declare_parameter("publish_depth_image", False)
        self.declare_parameter("publish_colormap", True)

        # ---- topics ----
        self.declare_parameter("disparity_topic", "/camera/disparity")
        self.declare_parameter("detection_topic", "/det/persons")

        self.declare_parameter("out_z_topic", "/depth/person_z")
        self.declare_parameter("out_status_topic", "/depth/person_status")
        self.declare_parameter("out_detected_topic", "/depth/person_detected")
        self.declare_parameter("out_depth_valid_topic", "/depth/depth_valid")
        self.declare_parameter("out_in_zone_topic", "/depth/in_zone")
        self.declare_parameter("out_color_topic", "/depth/colormap")
        self.declare_parameter("out_depth_image_topic", "/depth/image_m")

        # ---- monitored zone in image coordinates ----
        self.declare_parameter("zone_x1", 320)
        self.declare_parameter("zone_y1", 160)
        self.declare_parameter("zone_x2", 960)
        self.declare_parameter("zone_y2", 920)
        # ---- load params ----
        self.disp_scale = float(self.get_parameter("disp_scale").value)
        self.baseline_m = float(self.get_parameter("baseline_m").value)
        self.fx_px = float(self.get_parameter("fx_px").value)
        self.fx_reference_width_px = float(self.get_parameter("fx_reference_width_px").value)
        self.zmin = float(self.get_parameter("zmin").value)
        self.zmax = float(self.get_parameter("zmax").value)

        self.show_debug = bool(self.get_parameter("show_debug").value)
        self.min_valid_depth_pixels = int(self.get_parameter("min_valid_depth_pixels").value)
        self.depth_percentile = float(self.get_parameter("depth_percentile").value)
        self.depth_stat = str(self.get_parameter("depth_stat").value).strip().lower()
        self.mad_gate_sigma = float(self.get_parameter("mad_gate_sigma").value)
        self.max_det_age_ms = int(self.get_parameter("max_det_age_ms").value)
        self.smooth_depth = bool(self.get_parameter("smooth_depth").value)
        self.z_alpha = float(self.get_parameter("z_alpha").value)
        self.roi_mode = str(self.get_parameter("roi_mode").value).strip().lower()
        self.torso_side_frac = float(self.get_parameter("torso_side_frac").value)
        self.torso_top_frac = float(self.get_parameter("torso_top_frac").value)
        self.torso_bottom_frac = float(self.get_parameter("torso_bottom_frac").value)
        self.publish_depth_image = bool(self.get_parameter("publish_depth_image").value)
        self.publish_colormap = bool(self.get_parameter("publish_colormap").value)

        self.disparity_topic = str(self.get_parameter("disparity_topic").value)
        self.detection_topic = str(self.get_parameter("detection_topic").value)

        self.out_z_topic = str(self.get_parameter("out_z_topic").value)
        self.out_status_topic = str(self.get_parameter("out_status_topic").value)
        self.out_detected_topic = str(self.get_parameter("out_detected_topic").value)
        self.out_depth_valid_topic = str(self.get_parameter("out_depth_valid_topic").value)
        self.out_in_zone_topic = str(self.get_parameter("out_in_zone_topic").value)
        self.out_color_topic = str(self.get_parameter("out_color_topic").value)
        self.out_depth_image_topic = str(self.get_parameter("out_depth_image_topic").value)

        self.zone_x1 = int(self.get_parameter("zone_x1").value)
        self.zone_y1 = int(self.get_parameter("zone_y1").value)
        self.zone_x2 = int(self.get_parameter("zone_x2").value)
        self.zone_y2 = int(self.get_parameter("zone_y2").value)

        # ---- sanity checks ----
        if self.baseline_m <= 0.0:
            raise ValueError("baseline_m must be > 0")
        if self.fx_px <= 0.0:
            raise ValueError("fx_px must be > 0")
        if self.disp_scale <= 0.0:
            raise ValueError("disp_scale must be > 0")
        if self.zmin <= 0.0 or self.zmax <= 0.0 or self.zmin >= self.zmax:
            raise ValueError("Require 0 < zmin < zmax")
        if self.depth_stat not in ("median", "percentile"):
            self.get_logger().warn(f"Unknown depth_stat='{self.depth_stat}', falling back to 'median'")
            self.depth_stat = "median"
        if self.roi_mode not in ("full", "torso"):
            self.get_logger().warn(f"Unknown roi_mode='{self.roi_mode}', falling back to 'full'")
            self.roi_mode = "full"

        self.bridge = CvBridge()

        # latest detection state
        self.latest_bbox = None
        self.latest_det_count = 0
        self.latest_det_stamp_ns = None

        self.frame_i = 0
        self.last_z_filtered = None

        # ---- subscriptions ----
        self.sub_disp = self.create_subscription(Image, self.disparity_topic, self.on_disparity, qos_profile_sensor_data)
        self.sub_det = self.create_subscription(Detection2DArray, self.detection_topic, self.on_detections, qos_profile_sensor_data)

        # ---- publishers ----
        self.pub_z = self.create_publisher(Float32, self.out_z_topic, qos_profile_sensor_data)
        self.pub_status = self.create_publisher(String, self.out_status_topic, qos_profile_sensor_data)
        self.pub_detected = self.create_publisher(Bool, self.out_detected_topic, qos_profile_sensor_data)
        self.pub_depth_valid = self.create_publisher(Bool, self.out_depth_valid_topic, qos_profile_sensor_data)
        self.pub_in_zone = self.create_publisher(Bool, self.out_in_zone_topic, qos_profile_sensor_data)
        self.pub_color = self.create_publisher(Image, self.out_color_topic, qos_profile_sensor_data)
        self.pub_depth_image = self.create_publisher(Image, self.out_depth_image_topic, qos_profile_sensor_data)

        self.get_logger().info(
            f"Depth node started | fx_px={self.fx_px:.2f} baseline={self.baseline_m:.6f} "
            f"| roi_mode={self.roi_mode} depth_stat={self.depth_stat} "
            f"| zone=({self.zone_x1},{self.zone_y1})-({self.zone_x2},{self.zone_y2})"
        )

    # ---------- helpers ----------

    def build_roi(self, cx, cy, bw, bh):
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        if self.roi_mode == "torso":
            # Prefer torso center to reduce arm/leg/background disparity outliers.
            side = max(0.0, min(0.45, self.torso_side_frac))
            top = max(0.0, min(0.95, self.torso_top_frac))
            bottom = max(top + 0.02, min(1.0, self.torso_bottom_frac))

            tx1 = x1 + side * bw
            tx2 = x2 - side * bw
            ty1 = y1 + top * bh
            ty2 = y1 + bottom * bh

            if (tx2 - tx1) >= 8.0 and (ty2 - ty1) >= 8.0:
                return (tx1, ty1, tx2, ty2)

        return (x1, y1, x2, y2)

    def disparity_to_depth_m(self, disp_u16: np.ndarray, fx_eff: float) -> np.ndarray:
        d = disp_u16.astype(np.float32) / self.disp_scale
        d[d <= 0.0] = np.nan
        z = (fx_eff * self.baseline_m) / d
        return z.astype(np.float32)

    def depth_to_colormap(self, depth_m: np.ndarray) -> np.ndarray:
        d = depth_m.copy()
        d[~np.isfinite(d)] = self.zmax
        d = np.clip(d, self.zmin, self.zmax)
        norm = ((d - self.zmin) * (255.0 / max(1e-6, (self.zmax - self.zmin)))).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

    def clamp_bbox(self, bbox, w, h):
        x1, y1, x2, y2 = bbox
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w, round(x2))))
        y2 = int(max(0, min(h, round(y2))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def boxes_intersect(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)

    def publish_bool(self, pub, value: bool):
        msg = Bool()
        msg.data = bool(value)
        pub.publish(msg)

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)

    def detection_is_fresh(self):
        if self.latest_bbox is None or self.latest_det_stamp_ns is None:
            return False
        now_ns = self.get_clock().now().nanoseconds
        age_ms = (now_ns - self.latest_det_stamp_ns) / 1e6
        return age_ms <= self.max_det_age_ms

    def filter_depth_values(self, vals: np.ndarray) -> np.ndarray:
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= self.zmin) & (vals <= self.zmax)]
        return vals

    def estimate_depth(self, vals: np.ndarray):
        vals = self.filter_depth_values(vals)
        if vals.size < self.min_valid_depth_pixels:
            return float("nan"), False, vals

        if self.mad_gate_sigma > 0.0 and vals.size >= 5:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            robust_sigma = 1.4826 * mad
            if robust_sigma > 1e-6:
                gate = self.mad_gate_sigma * robust_sigma
                vals = vals[np.abs(vals - med) <= gate]
                if vals.size < self.min_valid_depth_pixels:
                    return float("nan"), False, vals

        if self.depth_stat == "percentile":
            z = float(np.percentile(vals, self.depth_percentile))
        else:
            z = float(np.median(vals))
        return z, True, vals

    # ---------- callbacks ----------

    def on_detections(self, msg: Detection2DArray):
        self.latest_det_count = len(msg.detections)
        self.latest_det_stamp_ns = self.get_clock().now().nanoseconds

        if len(msg.detections) == 0:
            self.latest_bbox = None
            return

        best_det = None
        best_area = -1.0

        for det in msg.detections:
            bw = float(det.bbox.size_x)
            bh = float(det.bbox.size_y)

            if bw < 10 or bh < 10:
                continue

            area = bw * bh

            if area > best_area:
                best_area = area
                best_det = det

        if best_det is None:
            self.latest_bbox = None
            return

        cx = float(best_det.bbox.center.position.x)
        cy = float(best_det.bbox.center.position.y)
        bw = float(best_det.bbox.size_x)
        bh = float(best_det.bbox.size_y)

        if bw < 10.0 or bh < 10.0:
            self.latest_bbox = None
            return

        self.latest_bbox = self.build_roi(cx, cy, bw, bh)
    def on_disparity(self, msg: Image):
        self.frame_i += 1

        try:
            disp = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"cv_bridge disp decode failed: {e}")
            return

        if disp.dtype != np.uint16:
            self.get_logger().warn(f"Expected uint16 disparity, got {disp.dtype}")
            return

        h, w = disp.shape[:2]
        fx_eff = self.fx_px
        if self.fx_reference_width_px > 0.0:
            fx_eff = self.fx_px * (float(w) / self.fx_reference_width_px)

        person_detected = False
        person_roi = None

        if self.detection_is_fresh():
            roi = self.clamp_bbox(self.latest_bbox, w, h)
            if roi is not None:
                person_detected = True
                person_roi = roi

        zone_roi = self.clamp_bbox((self.zone_x1, self.zone_y1, self.zone_x2, self.zone_y2), w, h)

        in_zone = False
        if person_roi is not None and zone_roi is not None:
            in_zone = self.boxes_intersect(person_roi, zone_roi)

        z = float("nan")
        depth_valid = False
        finite_count = 0
        total_count = 0

        if person_roi is not None:
            x1, y1, x2, y2 = person_roi
            patch_u16 = disp[y1:y2, x1:x2]
            patch = self.disparity_to_depth_m(patch_u16, fx_eff)
            total_count = int(patch.size)

            z, depth_valid, vals = self.estimate_depth(patch.reshape(-1))
            finite_count = int(vals.size)

            if depth_valid:
                if self.smooth_depth:
                    if self.last_z_filtered is None or not math.isfinite(self.last_z_filtered):
                        self.last_z_filtered = z
                    else:
                        self.last_z_filtered = self.z_alpha * self.last_z_filtered + (1.0 - self.z_alpha) * z
                    z = float(self.last_z_filtered)
            else:
                self.last_z_filtered = None
        else:
            self.last_z_filtered = None

        out_z = Float32()
        out_z.data = z if depth_valid and math.isfinite(z) else -1.0
        self.pub_z.publish(out_z)

        self.publish_bool(self.pub_detected, person_detected)
        self.publish_bool(self.pub_depth_valid, depth_valid)
        self.publish_bool(self.pub_in_zone, in_zone)

        status = (
            f"detected={person_detected} | in_zone={in_zone} | depth_valid={depth_valid} | "
            f"z={'%.2f m' % z if depth_valid and math.isfinite(z) else 'invalid'} | "
            f"valid={finite_count}/{total_count} | roi_mode={self.roi_mode}"
        )
        self.publish_status(status)

        need_depth_image = self.publish_depth_image and self.pub_depth_image.get_subscription_count() > 0
        need_colormap = self.publish_colormap and self.show_debug and self.pub_color.get_subscription_count() > 0

        if need_depth_image or need_colormap:
            depth_m = self.disparity_to_depth_m(disp, fx_eff)

            if need_depth_image:
                try:
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_m.astype(np.float32), encoding="32FC1")
                    depth_msg.header = msg.header
                    self.pub_depth_image.publish(depth_msg)
                except Exception as e:
                    self.get_logger().error(f"publish depth image failed: {e}")

        if need_colormap:
            color = self.depth_to_colormap(depth_m)

            if zone_roi is not None:
                zx1, zy1, zx2, zy2 = zone_roi
                cv2.rectangle(color, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)

            if person_roi is not None:
                x1, y1, x2, y2 = person_roi
                cv2.rectangle(color, (x1, y1), (x2, y2), (255, 255, 255), 2)

            label = f"det={person_detected} zone={in_zone} depth={depth_valid}"
            if depth_valid and math.isfinite(z):
                label += f" z={z:.2f}m"
            else:
                label += " z=invalid"

            cv2.putText(
                color, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            try:
                color_msg = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
                color_msg.header = msg.header
                self.pub_color.publish(color_msg)
            except Exception as e:
                self.get_logger().error(f"publish colormap failed: {e}")

        if (self.frame_i % 30) == 0:
            self.get_logger().info(status)


def main():
    rclpy.init()
    node = DepthNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
