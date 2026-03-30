import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
JSON_EXTS = {".json", ".geojson"}
MEASUREMENT_METHODS = ("mic", "min_feret")
METHOD_COLORS = {
    "mic": (0, 0, 255),
    "min_feret": (255, 0, 0),
}
METHOD_SHORT = {
    "mic": "MIC",
    "min_feret": "MinF",
}


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        mask = np.array(Image.open(path))
    if mask is None:
        raise ValueError(f"Could not read input file: {path}")

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            alpha = mask[..., 3]
            rgb = mask[..., :3]
            if np.any(alpha) and not np.any(rgb):
                mask = alpha
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return np.squeeze(mask)


def threshold_mask(mask, threshold):
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)

    if np.issubdtype(mask.dtype, np.floating):
        mask = np.nan_to_num(mask, nan=0.0)

    max_val = float(np.max(mask)) if mask.size else 0.0
    if max_val <= 1.0:
        return (mask > 0).astype(np.uint8)

    return (mask > threshold).astype(np.uint8)


def extract_contours(binary_mask, min_area):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or contour.shape[0] < 3:
            continue
        out.append(contour.reshape(-1, 2).astype(np.float32))
    return out


def looks_like_point_sequence(obj):
    if not isinstance(obj, list) or len(obj) < 3:
        return False
    return all(
        isinstance(pt, (list, tuple)) and len(pt) >= 2 and all(isinstance(v, (int, float)) for v in pt[:2])
        for pt in obj
    )


def extract_polygons_from_json_obj(obj, polygons):
    if isinstance(obj, dict):
        obj_type = obj.get("type")
        if obj_type == "FeatureCollection":
            for feature in obj.get("features", []):
                extract_polygons_from_json_obj(feature, polygons)
            return
        if obj_type == "Feature":
            extract_polygons_from_json_obj(obj.get("geometry"), polygons)
            return
        if obj_type == "Polygon":
            rings = obj.get("coordinates", [])
            if rings:
                polygons.append(np.asarray(rings[0], dtype=np.float32))
            return
        if obj_type == "MultiPolygon":
            for poly in obj.get("coordinates", []):
                if poly:
                    polygons.append(np.asarray(poly[0], dtype=np.float32))
            return
        if "polygons" in obj:
            extract_polygons_from_json_obj(obj["polygons"], polygons)
            return
        for value in obj.values():
            extract_polygons_from_json_obj(value, polygons)
        return

    if isinstance(obj, list):
        if looks_like_point_sequence(obj):
            polygons.append(np.asarray(obj, dtype=np.float32))
            return
        for item in obj:
            extract_polygons_from_json_obj(item, polygons)


def load_polygons_from_json(path, min_area):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygons = []
    extract_polygons_from_json_obj(data, polygons)

    filtered = []
    for polygon in polygons:
        polygon = ensure_closed_polygon(polygon)
        if polygon.shape[0] < 4:
            continue
        area = abs(polygon_area(polygon[:-1]))
        if area >= min_area:
            filtered.append(polygon[:-1].astype(np.float32))
    return filtered


def ensure_closed_polygon(points):
    if points.shape[0] == 0:
        return points
    if np.allclose(points[0], points[-1]):
        return points
    return np.vstack([points, points[0]])


def polygon_area(points):
    pts = ensure_closed_polygon(points)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def polygon_centroid(points):
    pts = ensure_closed_polygon(points)
    cross = pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]
    area_factor = np.sum(cross)
    if abs(area_factor) < 1e-8:
        return np.mean(points, axis=0).astype(np.float32)
    cx = np.sum((pts[:-1, 0] + pts[1:, 0]) * cross) / (3.0 * area_factor)
    cy = np.sum((pts[:-1, 1] + pts[1:, 1]) * cross) / (3.0 * area_factor)
    return np.array([cx, cy], dtype=np.float32)


def farthest_pair(points):
    best_dist = 0.0
    best_pair = (points[0], points[0])
    for i in range(len(points)):
        deltas = points[i + 1:] - points[i]
        if deltas.size == 0:
            continue
        distances = np.linalg.norm(deltas, axis=1)
        if distances.size == 0:
            continue
        idx = int(np.argmax(distances))
        dist = float(distances[idx])
        if dist > best_dist:
            best_dist = dist
            best_pair = (points[i], points[i + 1 + idx])
    return best_dist, best_pair


def min_feret_metrics(points):
    contour = points.reshape(-1, 1, 2).astype(np.float32)
    rect = cv2.minAreaRect(contour)
    center = np.array(rect[0], dtype=np.float32)
    width, height = rect[1]
    angle_deg = rect[2]

    if width <= 0 or height <= 0:
        return 0.0, (center, center)

    short_side = min(width, height)
    angle_rad = math.radians(angle_deg)
    edge_dir = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
    normal_dir = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float32)

    if width <= height:
        direction = edge_dir
    else:
        direction = normal_dir

    p1 = center - direction * (short_side / 2.0)
    p2 = center + direction * (short_side / 2.0)
    return float(short_side), (p1, p2)


def rasterize_polygon(points, padding=4):
    min_xy = np.floor(np.min(points, axis=0)).astype(int)
    max_xy = np.ceil(np.max(points, axis=0)).astype(int)
    size = max_xy - min_xy + 1 + 2 * padding
    size = np.maximum(size, 8)

    shifted = points - min_xy + padding
    canvas = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    poly = np.round(shifted).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [poly], 255)
    return canvas, shifted, min_xy, padding


def max_inscribed_circle(points):
    mask, _, min_xy, padding = rasterize_polygon(points)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_val = float(np.max(dist))
    if max_val <= 0:
        center = np.mean(points, axis=0)
        return 0.0, center, 0.0

    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    center = np.array([x, y], dtype=np.float32) + min_xy - padding
    return 2.0 * max_val, center, max_val


def sample_mic_instances(points, max_instances=8, spacing_factor=1.2):
    mask, _, min_xy, padding = rasterize_polygon(points)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    work = dist.copy()
    samples = []
    for instance_id in range(max_instances):
        max_val = float(np.max(work))
        if max_val <= 0:
            break
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        center = np.array([x, y], dtype=np.float32) + min_xy - padding
        samples.append(
            {
                "instance_id": instance_id + 1,
                "diameter_px": 2.0 * max_val,
                "anchor_x": float(center[0]),
                "anchor_y": float(center[1]),
            }
        )
        suppression_radius = max(2, int(round(max_val * spacing_factor)))
        cv2.circle(work, (int(x), int(y)), suppression_radius, 0.0, thickness=-1)
    return samples


def sample_min_feret_instances(points, max_instances=16):
    contour = points.reshape(-1, 1, 2).astype(np.float32)
    hull = cv2.convexHull(contour).reshape(-1, 2)
    if hull.shape[0] < 2:
        return []

    samples = []
    seen = set()
    centroid = polygon_centroid(points)
    for i in range(hull.shape[0]):
        p1 = hull[i]
        p2 = hull[(i + 1) % hull.shape[0]]
        edge = p2 - p1
        length = float(np.linalg.norm(edge))
        if length <= 1e-6:
            continue
        normal = np.array([-edge[1], edge[0]], dtype=np.float32) / length
        angle = round(float(math.degrees(math.atan2(normal[1], normal[0])) % 180.0), 3)
        if angle in seen:
            continue
        seen.add(angle)
        projections = hull @ normal
        width = float(np.max(projections) - np.min(projections))
        samples.append(
            {
                "diameter_px": width,
                "anchor_x": float(centroid[0]),
                "anchor_y": float(centroid[1]),
                "angle_deg": angle,
            }
        )
    samples.sort(key=lambda sample: sample["diameter_px"])
    trimmed = samples[:max_instances]
    for idx, sample in enumerate(trimmed, start=1):
        sample["instance_id"] = idx
    return trimmed


def sample_method_instances(row, method, max_instances, mic_spacing_factor=1.2):
    if method == "mic":
        return sample_mic_instances(
            row["contour"],
            max_instances=max_instances,
            spacing_factor=mic_spacing_factor,
        )
    if method == "min_feret":
        return sample_min_feret_instances(row["contour"], max_instances=max_instances)
    raise ValueError(f"Unsupported method: {method}")


def compute_metrics(points, method):
    points = points.astype(np.float32)
    contour = points.reshape(-1, 1, 2)
    hull = cv2.convexHull(contour).reshape(-1, 2)

    area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    centroid = polygon_centroid(points)
    min_feret, min_pair = min_feret_metrics(hull)
    mic_diameter, mic_center, mic_radius = max_inscribed_circle(points)

    if method == "mic":
        chosen = mic_diameter
    elif method == "min_feret":
        chosen = min_feret
    else:
        raise ValueError(f"Unsupported method: {method}")

    return {
        "area_px2": area,
        "perimeter_px": perimeter,
        "min_feret_diameter_px": min_feret,
        "mic_diameter_px": mic_diameter,
        "chosen_diameter_px": chosen,
        "min_feret_p1_x": float(min_pair[0][0]),
        "min_feret_p1_y": float(min_pair[0][1]),
        "min_feret_p2_x": float(min_pair[1][0]),
        "min_feret_p2_y": float(min_pair[1][1]),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "mic_center_x": float(mic_center[0]),
        "mic_center_y": float(mic_center[1]),
        "mic_radius_px": float(mic_radius),
        "contour": points,
    }


def iter_inputs(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in IMAGE_EXTS.union(JSON_EXTS):
                files.append(child)
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def resolve_methods(methods=None, default_method="mic"):
    if methods is None:
        return [default_method]
    if isinstance(methods, str):
        items = [item.strip() for item in methods.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in methods if str(item).strip()]
    if not items:
        return [default_method]
    if len(items) == 1 and items[0].lower() == "all":
        return list(MEASUREMENT_METHODS)
    normalized = []
    for item in items:
        lowered = item.lower()
        if lowered not in MEASUREMENT_METHODS:
            raise ValueError(f"Unsupported measurement method: {item}")
        if lowered not in normalized:
            normalized.append(lowered)
    return normalized


def measurement_value(row, method):
    if method == "mic":
        return row["mic_diameter_px"]
    if method == "min_feret":
        return row["min_feret_diameter_px"]
    raise ValueError(f"Unsupported method: {method}")


def build_measurement_rows(rows, pixel_size, unit_name, methods):
    out_rows = []
    for row in rows:
        for method in methods:
            out_rows.append(
                {
                    "source_file": row["source_file"],
                    "polygon_id": row["polygon_id"],
                    "measurement_method": method,
                    "diameter_px": measurement_value(row, method),
                    "diameter_units": measurement_value(row, method) * pixel_size,
                    "pixel_size": pixel_size,
                    "unit": unit_name,
                    "area_px2": row["area_px2"],
                    "area_units2": row["area_px2"] * (pixel_size ** 2),
                    "perimeter_px": row["perimeter_px"],
                    "perimeter_units": row["perimeter_px"] * pixel_size,
                }
            )
    return out_rows


def build_measurement_instances(rows, pixel_size, unit_name, methods, max_instances_per_method=8, mic_spacing_factor=1.2):
    instances = []
    for row in rows:
        for method in methods:
            for sample in sample_method_instances(
                row,
                method,
                max_instances=max_instances_per_method,
                mic_spacing_factor=mic_spacing_factor,
            ):
                instances.append(
                    {
                        "source_file": row["source_file"],
                        "polygon_id": row["polygon_id"],
                        "measurement_method": method,
                        "instance_id": sample["instance_id"],
                        "diameter_px": sample["diameter_px"],
                        "diameter_units": sample["diameter_px"] * pixel_size,
                        "pixel_size": pixel_size,
                        "unit": unit_name,
                        "anchor_x": sample.get("anchor_x"),
                        "anchor_y": sample.get("anchor_y"),
                        "angle_deg": sample.get("angle_deg"),
                    }
                )
    return instances


def summarize_instances(instances, group_keys):
    grouped = defaultdict(list)
    for row in instances:
        key = tuple(row[k] for k in group_keys)
        grouped[key].append(row["diameter_px"])

    out = []
    for key, values in grouped.items():
        values_np = np.asarray(values, dtype=np.float64)
        record = {group_keys[i]: key[i] for i in range(len(group_keys))}
        record["n_instances"] = int(values_np.size)
        record["diameter_mean_px"] = float(np.mean(values_np))
        record["diameter_std_px"] = float(np.std(values_np, ddof=0))
        record["diameter_min_px"] = float(np.min(values_np))
        record["diameter_max_px"] = float(np.max(values_np))
        out.append(record)
    return out


def write_csv_rows(rows, out_csv, fieldnames):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def serialize_measurement_rows(rows, out_csv, pixel_size, unit_name, methods):
    fieldnames = [
        "source_file",
        "polygon_id",
        "measurement_method",
        "diameter_px",
        "diameter_units",
        "pixel_size",
        "unit",
        "area_px2",
        "area_units2",
        "perimeter_px",
        "perimeter_units",
    ]
    measurement_rows = build_measurement_rows(rows, pixel_size, unit_name, methods)
    write_csv_rows(measurement_rows, out_csv, fieldnames)


def serialize_measurement_instances(rows, out_csv, pixel_size, unit_name, methods, max_instances_per_method=8, mic_spacing_factor=1.2):
    fieldnames = [
        "source_file",
        "polygon_id",
        "measurement_method",
        "instance_id",
        "diameter_px",
        "diameter_units",
        "pixel_size",
        "unit",
        "anchor_x",
        "anchor_y",
        "angle_deg",
    ]
    instances = build_measurement_instances(
        rows,
        pixel_size,
        unit_name,
        methods,
        max_instances_per_method=max_instances_per_method,
        mic_spacing_factor=mic_spacing_factor,
    )
    write_csv_rows(instances, out_csv, fieldnames)
    return instances


def serialize_instance_summaries(instances, per_polygon_csv, per_image_csv, pixel_size, unit_name):
    polygon_rows = summarize_instances(instances, ["source_file", "polygon_id", "measurement_method"])
    image_rows = summarize_instances(instances, ["source_file", "measurement_method"])

    for rows in (polygon_rows, image_rows):
        for row in rows:
            row["pixel_size"] = pixel_size
            row["unit"] = unit_name
            row["diameter_mean_units"] = row["diameter_mean_px"] * pixel_size
            row["diameter_std_units"] = row["diameter_std_px"] * pixel_size
            row["diameter_min_units"] = row["diameter_min_px"] * pixel_size
            row["diameter_max_units"] = row["diameter_max_px"] * pixel_size

    polygon_fields = [
        "source_file",
        "polygon_id",
        "measurement_method",
        "n_instances",
        "diameter_mean_px",
        "diameter_std_px",
        "diameter_min_px",
        "diameter_max_px",
        "pixel_size",
        "unit",
        "diameter_mean_units",
        "diameter_std_units",
        "diameter_min_units",
        "diameter_max_units",
    ]
    image_fields = [
        "source_file",
        "measurement_method",
        "n_instances",
        "diameter_mean_px",
        "diameter_std_px",
        "diameter_min_px",
        "diameter_max_px",
        "pixel_size",
        "unit",
        "diameter_mean_units",
        "diameter_std_units",
        "diameter_min_units",
        "diameter_max_units",
    ]
    write_csv_rows(polygon_rows, per_polygon_csv, polygon_fields)
    write_csv_rows(image_rows, per_image_csv, image_fields)


def draw_measurement(canvas, method, sample):
    color = METHOD_COLORS[method]
    if method == "mic":
        center = (int(round(sample["anchor_x"])), int(round(sample["anchor_y"])))
        radius = max(1, int(round(sample["diameter_px"] / 2.0)))
        cv2.circle(canvas, center, radius, color, 1)
    elif method == "min_feret":
        center = np.array([sample["anchor_x"], sample["anchor_y"]], dtype=np.float32)
        angle_rad = math.radians(float(sample.get("angle_deg", 0.0)))
        direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
        p1 = center - direction * (sample["diameter_px"] / 2.0)
        p2 = center + direction * (sample["diameter_px"] / 2.0)
        cv2.line(
            canvas,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            color,
            1,
        )


def draw_measurements(canvas, row, methods, show_values=True, max_instances_per_method=3, mic_spacing_factor=1.2):
    contour = np.round(row["contour"]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [contour], True, (0, 255, 0), 1)
    label_parts = []
    for method in methods:
        samples = sample_method_instances(
            row,
            method,
            max_instances=max_instances_per_method,
            mic_spacing_factor=mic_spacing_factor,
        )
        for sample in samples:
            draw_measurement(canvas, method, sample)
        if show_values and samples:
            values = np.asarray([sample["diameter_px"] for sample in samples], dtype=np.float64)
            label_parts.append(f'{METHOD_SHORT[method]}:{np.mean(values):.1f}+/-{np.std(values):.1f}')

    label_anchor = tuple(np.round(row["contour"][0]).astype(int))
    if show_values:
        label = f'{row["polygon_id"]} ' + " ".join(label_parts)
    else:
        label = str(row["polygon_id"])
    cv2.putText(canvas, label, label_anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)


def save_overlay(path, rows, methods, overlay_dir, threshold, show_values=True, max_instances_per_method=3, mic_spacing_factor=1.2):
    mask = load_mask(path)
    binary = threshold_mask(mask, threshold)
    canvas = (binary * 255).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for row in rows:
        draw_measurements(
            canvas,
            row,
            methods,
            show_values=show_values,
            max_instances_per_method=max_instances_per_method,
            mic_spacing_factor=mic_spacing_factor,
        )

    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_path = overlay_dir / f"{path.stem}_diameters.png"
    cv2.imwrite(str(out_path), canvas)


def serialize_rows(rows, out_csv, pixel_size, unit_name, method):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "polygon_id",
        "method",
        "pixel_size",
        "unit",
        "area_px2",
        "perimeter_px",
        "min_feret_diameter_px",
        "mic_diameter_px",
        "chosen_diameter_px",
        "area_units2",
        "perimeter_units",
        "min_feret_diameter_units",
        "mic_diameter_units",
        "chosen_diameter_units",
        "min_feret_p1_x",
        "min_feret_p1_y",
        "min_feret_p2_x",
        "min_feret_p2_y",
        "centroid_x",
        "centroid_y",
        "mic_center_x",
        "mic_center_y",
        "mic_radius_px",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            export = {k: v for k, v in row.items() if k != "contour"}
            export["method"] = method
            export["pixel_size"] = pixel_size
            export["unit"] = unit_name
            export["area_units2"] = row["area_px2"] * (pixel_size ** 2)
            export["perimeter_units"] = row["perimeter_px"] * pixel_size
            export["min_feret_diameter_units"] = row["min_feret_diameter_px"] * pixel_size
            export["mic_diameter_units"] = row["mic_diameter_px"] * pixel_size
            export["chosen_diameter_units"] = row["chosen_diameter_px"] * pixel_size
            writer.writerow(export)


def process_path(path, method, threshold, min_area):
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        mask = load_mask(path)
        binary = threshold_mask(mask, threshold)
        return extract_contours(binary, min_area)
    if suffix in JSON_EXTS:
        return load_polygons_from_json(path, min_area)
    raise ValueError(f"Unsupported file extension: {path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract robust diameter measurements from SAM mask polygons or polygon JSON/GeoJSON."
    )
    parser.add_argument("--input", required=True, help="Mask image, JSON/GeoJSON file, or a directory containing them.")
    parser.add_argument("--output_csv", default="sam_polygon_diameters.csv", help="Path to the output CSV.")
    parser.add_argument(
        "--method",
        choices=["mic", "min_feret"],
        default="mic",
        help="Diameter metric used for the main chosen_diameter columns. Default: mic.",
    )
    parser.add_argument("--threshold", type=float, default=127.0, help="Binary threshold for mask images.")
    parser.add_argument("--min_area", type=float, default=16.0, help="Ignore polygons smaller than this area in pixels.")
    parser.add_argument("--pixel_size", type=float, default=1.0, help="Physical size of one pixel.")
    parser.add_argument("--unit", default="px", help="Label used for physical-size output columns.")
    parser.add_argument("--overlay_dir", help="Optional directory for diagnostic overlay PNGs.")
    parser.add_argument("--methods", default=None, help="Comma-separated measurement methods or 'all'. Example: mic,min_feret")
    parser.add_argument("--measurements_csv", help="Optional long-format CSV with one row per polygon per measurement method.")
    parser.add_argument("--instances_csv", help="Optional instance-level CSV with repeated measurements per polygon and method.")
    parser.add_argument("--per_polygon_stats_csv", help="Optional summary CSV with mean/std per polygon and method.")
    parser.add_argument("--per_image_stats_csv", help="Optional summary CSV with mean/std per image and method.")
    parser.add_argument("--max_instances_per_method", type=int, default=3, help="How many measurement geometries to draw and sample per method per polygon.")
    parser.add_argument("--mic_spacing_factor", type=float, default=1.2, help="How aggressively MIC circles suppress nearby candidates. Larger means more spacing.")
    parser.add_argument("--overlay-labels", choices=["values", "id"], default="values", help="Overlay labels as values or polygon id only.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    input_files = iter_inputs(args.input)
    if not input_files:
        raise ValueError(f"No supported files found in {args.input}")

    all_rows = []
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    methods = resolve_methods(args.methods, default_method=args.method)

    for path in input_files:
        polygons = process_path(path, args.method, args.threshold, args.min_area)
        source_rows = []
        for idx, points in enumerate(polygons, start=1):
            row = compute_metrics(points, args.method)
            row["source_file"] = str(path)
            row["polygon_id"] = idx
            source_rows.append(row)
            all_rows.append(row)

        if overlay_dir is not None and path.suffix.lower() in IMAGE_EXTS and source_rows:
            save_overlay(
                path,
                source_rows,
                methods,
                overlay_dir,
                args.threshold,
                show_values=(args.overlay_labels == "values"),
                max_instances_per_method=args.max_instances_per_method,
                mic_spacing_factor=args.mic_spacing_factor,
            )

    serialize_rows(all_rows, args.output_csv, args.pixel_size, args.unit, args.method)
    if args.measurements_csv:
        serialize_measurement_rows(all_rows, args.measurements_csv, args.pixel_size, args.unit, methods)
    instances_csv = Path(args.instances_csv) if args.instances_csv else None
    polygon_stats_csv = Path(args.per_polygon_stats_csv) if args.per_polygon_stats_csv else None
    image_stats_csv = Path(args.per_image_stats_csv) if args.per_image_stats_csv else None
    if instances_csv or polygon_stats_csv or image_stats_csv:
        instances = serialize_measurement_instances(
            all_rows,
            instances_csv or Path(args.output_csv).with_name("polygon_measurement_instances.csv"),
            args.pixel_size,
            args.unit,
            methods,
            max_instances_per_method=args.max_instances_per_method,
            mic_spacing_factor=args.mic_spacing_factor,
        )
        serialize_instance_summaries(
            instances,
            polygon_stats_csv or Path(args.output_csv).with_name("polygon_measurement_stats.csv"),
            image_stats_csv or Path(args.output_csv).with_name("image_measurement_stats.csv"),
            args.pixel_size,
            args.unit,
        )
    print(f"Processed {len(input_files)} file(s) and wrote {len(all_rows)} polygon measurements to {args.output_csv}")


if __name__ == "__main__":
    main()
