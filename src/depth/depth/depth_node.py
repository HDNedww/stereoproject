#!/usr/bin/env python3
import math
from collections import deque
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String, Bool
from vision_msgs.msg import Detection2DArray

from cv_bridge import CvBridge


class DepthNode(Node):
    def __init__(self):
        super().__init__("depth_node")

        # ---- stereo / depth params ----
        self.declare_parameter("disp_scale", 16.0)
        self.declare_parameter("baseline_m", 0.160)
        self.declare_parameter("fx_px", 540.73152)
        self.declare_parameter("fx_reference_width_px", 640)
        self.declare_parameter("zmin", 0.50)
        self.declare_parameter("zmax", 3.00)
        self.declare_parameter("auto_read_camera_params", False)
        self.declare_parameter("camera_device_id", "")
        self.declare_parameter("auto_override_baseline_m", True)
        self.declare_parameter("auto_override_fx_px", True)
        self.declare_parameter("auto_override_disp_scale", True)

        # ---- processing params ----
        self.declare_parameter("show_debug", True)
        self.declare_parameter("min_valid_depth_pixels", 6)
        self.declare_parameter("depth_percentile", 10.0)
        self.declare_parameter("depth_stat", "median")  # median | percentile
        self.declare_parameter("mad_gate_sigma", 2.5)   # 0 disables outlier rejection
        self.declare_parameter("max_det_age_ms", 300)
        self.declare_parameter("smooth_depth", True)
        self.declare_parameter("z_alpha", 0.7)
        self.declare_parameter("z_alpha_towards", 0.10)
        self.declare_parameter("z_alpha_away", 0.20)
        self.declare_parameter("temporal_window", 1)
        self.declare_parameter("max_jump_m", 0.20)
        self.declare_parameter("max_jump_towards_m", 0.80)
        self.declare_parameter("max_jump_away_m", 0.45)
        self.declare_parameter("reset_after_invalid_frames", 4)
        self.declare_parameter("hold_last_valid_frames", 6)
        self.declare_parameter("roi_mode", "full")  # full | torso
        self.declare_parameter("torso_side_frac", 0.20)
        self.declare_parameter("torso_top_frac", 0.15)
        self.declare_parameter("torso_bottom_frac", 0.88)
        self.declare_parameter("publish_depth_image", False)
        self.declare_parameter("publish_colormap", True)

        # ---- topics ----
        self.declare_parameter("disparity_topic", "/camera/disparity")
        self.declare_parameter("detection_topic", "/det/persons")
        self.declare_parameter("detection_image_topic", "/camera/intensity_rgb")
        self.declare_parameter("scale_detection_to_disparity", True)

        self.declare_parameter("out_z_topic", "/depth/person_z")
        self.declare_parameter("out_status_topic", "/depth/person_status")
        self.declare_parameter("out_detected_topic", "/depth/person_detected")
        self.declare_parameter("out_depth_valid_topic", "/depth/depth_valid")
        self.declare_parameter("out_in_zone_topic", "/depth/in_zone")
        self.declare_parameter("out_color_topic", "/depth/colormap")
        self.declare_parameter("out_depth_image_topic", "/depth/image_m")
        self.declare_parameter("qos_reliability", "best_effort")  # reliable | best_effort
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("use_zone", False)

        # ---- monitored zone in image coordinates ----
        self.declare_parameter("zone_use_percent", True)
        self.declare_parameter("zone_x1_pct", 0.25)
        self.declare_parameter("zone_y1_pct", 0.17)
        self.declare_parameter("zone_x2_pct", 0.75)
        self.declare_parameter("zone_y2_pct", 0.95)
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
        self.auto_read_camera_params = bool(self.get_parameter("auto_read_camera_params").value)
        self.camera_device_id = str(self.get_parameter("camera_device_id").value).strip()
        self.auto_override_baseline_m = bool(self.get_parameter("auto_override_baseline_m").value)
        self.auto_override_fx_px = bool(self.get_parameter("auto_override_fx_px").value)
        self.auto_override_disp_scale = bool(self.get_parameter("auto_override_disp_scale").value)

        self.show_debug = bool(self.get_parameter("show_debug").value)
        self.min_valid_depth_pixels = int(self.get_parameter("min_valid_depth_pixels").value)
        self.depth_percentile = float(self.get_parameter("depth_percentile").value)
        self.depth_stat = str(self.get_parameter("depth_stat").value).strip().lower()
        self.mad_gate_sigma = float(self.get_parameter("mad_gate_sigma").value)
        self.max_det_age_ms = int(self.get_parameter("max_det_age_ms").value)
        self.smooth_depth = bool(self.get_parameter("smooth_depth").value)
        self.z_alpha = float(self.get_parameter("z_alpha").value)
        self.z_alpha_towards = float(self.get_parameter("z_alpha_towards").value)
        self.z_alpha_away = float(self.get_parameter("z_alpha_away").value)
        self.temporal_window = int(self.get_parameter("temporal_window").value)
        self.max_jump_m = float(self.get_parameter("max_jump_m").value)
        self.max_jump_towards_m = float(self.get_parameter("max_jump_towards_m").value)
        self.max_jump_away_m = float(self.get_parameter("max_jump_away_m").value)
        self.reset_after_invalid_frames = int(self.get_parameter("reset_after_invalid_frames").value)
        self.hold_last_valid_frames = int(self.get_parameter("hold_last_valid_frames").value)
        self.roi_mode = str(self.get_parameter("roi_mode").value).strip().lower()
        self.torso_side_frac = float(self.get_parameter("torso_side_frac").value)
        self.torso_top_frac = float(self.get_parameter("torso_top_frac").value)
        self.torso_bottom_frac = float(self.get_parameter("torso_bottom_frac").value)
        self.publish_depth_image = bool(self.get_parameter("publish_depth_image").value)
        self.publish_colormap = bool(self.get_parameter("publish_colormap").value)

        self.disparity_topic = str(self.get_parameter("disparity_topic").value)
        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.detection_image_topic = str(self.get_parameter("detection_image_topic").value)
        self.scale_detection_to_disparity = bool(self.get_parameter("scale_detection_to_disparity").value)

        self.out_z_topic = str(self.get_parameter("out_z_topic").value)
        self.out_status_topic = str(self.get_parameter("out_status_topic").value)
        self.out_detected_topic = str(self.get_parameter("out_detected_topic").value)
        self.out_depth_valid_topic = str(self.get_parameter("out_depth_valid_topic").value)
        self.out_in_zone_topic = str(self.get_parameter("out_in_zone_topic").value)
        self.out_color_topic = str(self.get_parameter("out_color_topic").value)
        self.out_depth_image_topic = str(self.get_parameter("out_depth_image_topic").value)
        self.qos_reliability = str(self.get_parameter("qos_reliability").value).strip().lower()
        self.qos_depth = int(self.get_parameter("qos_depth").value)
        self.use_zone = bool(self.get_parameter("use_zone").value)

        self.zone_x1 = int(self.get_parameter("zone_x1").value)
        self.zone_y1 = int(self.get_parameter("zone_y1").value)
        self.zone_x2 = int(self.get_parameter("zone_x2").value)
        self.zone_y2 = int(self.get_parameter("zone_y2").value)
        self.zone_use_percent = bool(self.get_parameter("zone_use_percent").value)
        self.zone_x1_pct = float(self.get_parameter("zone_x1_pct").value)
        self.zone_y1_pct = float(self.get_parameter("zone_y1_pct").value)
        self.zone_x2_pct = float(self.get_parameter("zone_x2_pct").value)
        self.zone_y2_pct = float(self.get_parameter("zone_y2_pct").value)

        if self.auto_read_camera_params:
            self._try_load_camera_params()

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
        if self.temporal_window < 1:
            self.temporal_window = 1
        if self.reset_after_invalid_frames < 1:
            self.reset_after_invalid_frames = 1
        if self.hold_last_valid_frames < 0:
            self.hold_last_valid_frames = 0
        self.z_alpha_towards = min(0.99, max(0.0, self.z_alpha_towards))
        self.z_alpha_away = min(0.99, max(0.0, self.z_alpha_away))
        self.zone_x1_pct = min(1.0, max(0.0, self.zone_x1_pct))
        self.zone_y1_pct = min(1.0, max(0.0, self.zone_y1_pct))
        self.zone_x2_pct = min(1.0, max(0.0, self.zone_x2_pct))
        self.zone_y2_pct = min(1.0, max(0.0, self.zone_y2_pct))
        if self.zone_x2_pct <= self.zone_x1_pct:
            self.zone_x1_pct, self.zone_x2_pct = 0.25, 0.75
        if self.zone_y2_pct <= self.zone_y1_pct:
            self.zone_y1_pct, self.zone_y2_pct = 0.17, 0.95

        self.bridge = CvBridge()
        self.qos = self._build_qos()

        # latest detection state
        self.latest_bbox = None
        self.latest_bboxes = []
        self.latest_det_count = 0
        self.latest_det_stamp_ns = None
        self.latest_det_img_w = None
        self.latest_det_img_h = None
        self._scale_logged = False

        self.frame_i = 0
        self.last_z_filtered = None
        self.invalid_streak = 0
        self.z_history = deque(maxlen=self.temporal_window)
        self.last_good_z = None
        self.invalid_hold_count = 0
        self._zone_logged = False

        # ---- subscriptions ----
        self.sub_disp = self.create_subscription(Image, self.disparity_topic, self.on_disparity, self.qos)
        self.sub_det = self.create_subscription(Detection2DArray, self.detection_topic, self.on_detections, self.qos)
        self.sub_det_img = self.create_subscription(Image, self.detection_image_topic, self.on_detection_image, self.qos)

        # ---- publishers ----
        self.pub_z = self.create_publisher(Float32, self.out_z_topic, self.qos)
        self.pub_status = self.create_publisher(String, self.out_status_topic, self.qos)
        self.pub_detected = self.create_publisher(Bool, self.out_detected_topic, self.qos)
        self.pub_depth_valid = self.create_publisher(Bool, self.out_depth_valid_topic, self.qos)
        self.pub_in_zone = self.create_publisher(Bool, self.out_in_zone_topic, self.qos)
        self.pub_color = self.create_publisher(Image, self.out_color_topic, self.qos)
        self.pub_depth_image = self.create_publisher(Image, self.out_depth_image_topic, self.qos)

        self.get_logger().info(
            f"Depth node started | fx_px={self.fx_px:.2f} baseline={self.baseline_m:.6f} "
            f"| roi_mode={self.roi_mode} depth_stat={self.depth_stat} use_zone={self.use_zone}"
        )
        self.get_logger().info(
            f"Detection scaling: enabled={self.scale_detection_to_disparity} det_img_topic={self.detection_image_topic}"
        )
        self.get_logger().info(f"QoS: reliability={self.qos_reliability} depth={self.qos.depth}")

    # ---------- helpers ----------
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

    def _read_feature_float(self, dev, feature_name):
        try:
            return float(dev.get_float_feature_value(feature_name))
        except Exception:
            pass
        try:
            return float(dev.get_integer_feature_value(feature_name))
        except Exception:
            return None

    def _read_first_feature(self, dev, names):
        for n in names:
            v = self._read_feature_float(dev, n)
            if v is not None and math.isfinite(v):
                return v, n
        return None, None

    def _find_rc_visard_device_id(self, aravis):
        try:
            aravis.enable_interface("GigEVision")
            aravis.update_device_list()
            for i in range(aravis.get_n_devices()):
                dev_id = aravis.get_device_id(i)
                if "rc_visard" in str(dev_id).lower():
                    return str(dev_id)
        except Exception:
            return None
        return None

    def _try_load_camera_params(self):
        try:
            import gi
            gi.require_version("Aravis", "0.8")
            from gi.repository import Aravis
        except Exception as e:
            self.get_logger().warn(f"Auto camera params skipped (Aravis not available): {e}")
            return

        dev_id = self.camera_device_id if self.camera_device_id else self._find_rc_visard_device_id(Aravis)
        if not dev_id:
            self.get_logger().warn("Auto camera params skipped (rc_visard device not found)")
            return

        try:
            cam = Aravis.Camera.new(dev_id)
            dev = cam.get_device()
        except Exception as e:
            self.get_logger().warn(f"Auto camera params skipped (connect failed): {e}")
            return

        baseline_v, baseline_name = self._read_first_feature(
            dev, ["Scan3dBaseline", "Baseline", "Scan3dBaselineOffset"]
        )
        fx_v, fx_name = self._read_first_feature(
            dev, ["Scan3dFocalLength", "FocalLength", "Scan3dFocalLengthRaw"]
        )
        scale_v, scale_name = self._read_first_feature(
            dev, ["Scan3dCoordinateScale", "CoordinateScale", "DisparityScale"]
        )

        if self.auto_override_baseline_m and baseline_v is not None and baseline_v > 0.0:
            baseline_m = baseline_v * 1e-3 if baseline_v > 2.0 else baseline_v
            if 0.03 <= baseline_m <= 0.50:
                self.baseline_m = float(baseline_m)
                self.get_logger().info(f"Auto baseline_m={self.baseline_m:.6f} from feature '{baseline_name}'")
            else:
                self.get_logger().warn(
                    f"Ignoring baseline from '{baseline_name}' ({baseline_v}); outside expected range"
                )

        if self.auto_override_fx_px and fx_v is not None and fx_v > 20.0:
            if 100.0 <= fx_v <= 5000.0:
                self.fx_px = float(fx_v)
                self.get_logger().info(f"Auto fx_px={self.fx_px:.3f} from feature '{fx_name}'")
            else:
                self.get_logger().warn(
                    f"Ignoring focal from '{fx_name}' ({fx_v:.3f}); outside expected pixel range"
                )
        elif self.auto_override_fx_px and fx_v is not None:
            self.get_logger().warn(
                f"Feature '{fx_name}' returned {fx_v:.3f}; keeping fx_px={self.fx_px:.3f} (looks non-pixel)"
            )

        if self.auto_override_disp_scale and scale_v is not None and scale_v > 0.0:
            # Some cameras expose coordinate scale, others disparity scale.
            # Accept only values that map into realistic disparity quantization.
            cand = [float(scale_v)]
            if scale_v < 1.0:
                cand.append(float(1.0 / scale_v))
            accepted = None
            for v in cand:
                if 4.0 <= v <= 64.0:
                    accepted = v
                    break
            if accepted is not None:
                self.disp_scale = accepted
                self.get_logger().info(f"Auto disp_scale={self.disp_scale:.6f} from feature '{scale_name}'")
            else:
                self.get_logger().warn(
                    f"Ignoring disp scale from '{scale_name}' ({scale_v}); using disp_scale={self.disp_scale:.6f}"
                )

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

    def scale_bbox_to_disparity(self, bbox, disp_w, disp_h):
        if not self.scale_detection_to_disparity:
            return bbox
        if self.latest_det_img_w is None or self.latest_det_img_h is None:
            return bbox
        src_w = float(self.latest_det_img_w)
        src_h = float(self.latest_det_img_h)
        if src_w <= 1.0 or src_h <= 1.0:
            return bbox

        sx = float(disp_w) / src_w
        sy = float(disp_h) / src_h
        if abs(sx - 1.0) < 1e-3 and abs(sy - 1.0) < 1e-3:
            return bbox

        if not self._scale_logged:
            self.get_logger().info(
                f"Scaling det ROI from {int(src_w)}x{int(src_h)} to {disp_w}x{disp_h} (sx={sx:.3f}, sy={sy:.3f})"
            )
            self._scale_logged = True

        x1, y1, x2, y2 = bbox
        return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

    def resolve_zone_roi(self, w, h):
        if self.zone_use_percent:
            x1 = int(round(self.zone_x1_pct * w))
            y1 = int(round(self.zone_y1_pct * h))
            x2 = int(round(self.zone_x2_pct * w))
            y2 = int(round(self.zone_y2_pct * h))
        else:
            x1, y1, x2, y2 = self.zone_x1, self.zone_y1, self.zone_x2, self.zone_y2

        roi = self.clamp_bbox((x1, y1, x2, y2), w, h)
        if not self._zone_logged:
            mode = "percent" if self.zone_use_percent else "pixels"
            self.get_logger().info(f"Zone ROI mode={mode} resolved={roi} on {w}x{h}")
            self._zone_logged = True
        return roi

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
        if len(self.latest_bboxes) == 0 or self.latest_det_stamp_ns is None:
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

    def smooth_z(self, z_raw: float) -> float:
        self.z_history.append(float(z_raw))
        z_temporal = float(np.median(np.asarray(self.z_history, dtype=np.float32)))

        if self.last_z_filtered is None or not math.isfinite(self.last_z_filtered):
            self.last_z_filtered = z_temporal
            return z_temporal

        dz = z_temporal - self.last_z_filtered

        if dz < 0.0:
            jump_limit = self.max_jump_towards_m if self.max_jump_towards_m > 0.0 else self.max_jump_m
            alpha = self.z_alpha_towards
        else:
            jump_limit = self.max_jump_away_m if self.max_jump_away_m > 0.0 else self.max_jump_m
            alpha = self.z_alpha_away

        if jump_limit > 0.0 and abs(dz) > jump_limit:
            z_temporal = self.last_z_filtered + math.copysign(jump_limit, dz)

        self.last_z_filtered = alpha * self.last_z_filtered + (1.0 - alpha) * z_temporal
        return float(self.last_z_filtered)

    # ---------- callbacks ----------
    def on_detection_image(self, msg: Image):
        # Track detector input resolution so bbox coordinates can be mapped to disparity resolution.
        self.latest_det_img_w = int(msg.width)
        self.latest_det_img_h = int(msg.height)

    def on_detections(self, msg: Detection2DArray):
        self.latest_det_count = len(msg.detections)
        self.latest_det_stamp_ns = self.get_clock().now().nanoseconds

        if len(msg.detections) == 0:
            self.latest_bbox = None
            self.latest_bboxes = []
            return

        candidate_bboxes = []

        for det in msg.detections:
            bw = float(det.bbox.size_x)
            bh = float(det.bbox.size_y)

            if bw < 10 or bh < 10:
                continue

            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            candidate_bboxes.append(self.build_roi(cx, cy, bw, bh))

        self.latest_bboxes = candidate_bboxes
        self.latest_bbox = candidate_bboxes[0] if len(candidate_bboxes) > 0 else None
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
            person_detected = True
            best_z = float("inf")
            best_roi = None
            best_vals = None
            best_total_count = 0

            for bbox in self.latest_bboxes:
                bbox_scaled = self.scale_bbox_to_disparity(bbox, w, h)
                roi = self.clamp_bbox(bbox_scaled, w, h)
                if roi is None:
                    continue

                x1, y1, x2, y2 = roi
                patch_u16 = disp[y1:y2, x1:x2]
                patch = self.disparity_to_depth_m(patch_u16, fx_eff)
                z_candidate, valid_candidate, vals_candidate = self.estimate_depth(patch.reshape(-1))
                if not valid_candidate or not math.isfinite(z_candidate):
                    continue

                if z_candidate < best_z:
                    best_z = z_candidate
                    best_roi = roi
                    best_vals = vals_candidate
                    best_total_count = int(patch.size)

            person_roi = best_roi
            if person_roi is None:
                person_detected = False

        zone_roi = self.resolve_zone_roi(w, h) if self.use_zone else None

        if self.use_zone:
            in_zone = False
            if person_roi is not None and zone_roi is not None:
                in_zone = self.boxes_intersect(person_roi, zone_roi)
        else:
            in_zone = person_roi is not None

        z = float("nan")
        depth_valid = False
        finite_count = 0
        total_count = 0

        if person_roi is not None:
            if best_vals is not None:
                z = float(best_z)
                depth_valid = True
                finite_count = int(best_vals.size)
                total_count = int(best_total_count)

            if depth_valid:
                self.invalid_streak = 0
                self.invalid_hold_count = 0
                if self.smooth_depth:
                    z = self.smooth_z(z)
                self.last_good_z = z
            else:
                self.invalid_streak += 1
        else:
            self.invalid_streak += 1

        # Hide short dropouts: reuse last valid distance for a few frames.
        if (not depth_valid) and self.last_good_z is not None and self.invalid_hold_count < self.hold_last_valid_frames:
            z = float(self.last_good_z)
            depth_valid = True
            self.invalid_hold_count += 1

        if self.invalid_streak >= self.reset_after_invalid_frames:
            self.last_z_filtered = None
            self.z_history.clear()
            self.last_good_z = None
            self.invalid_hold_count = 0

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

            if self.use_zone and zone_roi is not None:
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
