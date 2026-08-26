#!/usr/bin/env python3
"""
kml_to_tiles_and_predict.py

- Read a KML polygon
- Tile the polygon's bounding box into 1024x1024 Google Static Maps tiles at zoom 19 (scale=2)
- Download tiles named kmz-zm-19_<lat>_<lon>.png
- Run prediction on each tile
- Restrict counted mask to polygon ROI (per-tile cropped)
- Save overlay PNG and kmz-zm-19_<lat>_<lon>.txt with area/panel estimates and TL/BR coords

Dependencies:
- requests
- PIL / pillow
- numpy
- matplotlib (for plt.imread)
- your local modules: model_list, data_processing.prepare_data, data_processing.data_processing_tool_4

"""

import os
import re
import math
import time
import requests
import datetime
import xml.etree.ElementTree as ET
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Import model & data helpers (adjust if your project structure differs)
from model_list import segnet_1, segnet_3, segnet_0, fast_scnn_2
import sys
sys.path.append("..")
from data_processing import prepare_data, data_processing_tool_4


# ----------------------------
# HARD-CODED CONFIG (edit)
# ----------------------------
API_KEY = ""   # <<-- Add an API_KEY
KML_PATH = "kml/lastest_all.kml"                          # KML file path
INPUT_IMAGE_DIR = "save_input"              # directory where tiles will be saved
SAVING_PATH = "save_output"                # outputs saved here
MODEL_PATH = "trained_models"
MODEL_TYPE = 1
MODEL_NAME = "fast_scnn_2.h5"
IMAGE_SIZE_PX = 1280
GMAPS_ZOOM = 19
GMAPS_SCALE = 2                               # must match how you want to request tiles (1 or 2)
GMAPS_MAPTYPE = "satellite"
CONF_THRESHOLD = 0.2
INITIAL_BATCH_SIZE = 8
PANEL_AREA_M2 = 2.57                          # area of a single panel in m^2
REQUEST_PAUSE_SECONDS = 0.2                   # small pause between map requests to be gentle on API
# ----------------------------


FILENAME_RE = re.compile(
    r"^kmz-zm-(?P<zoom>\d+)_(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+)\.png$",
    re.IGNORECASE
)


# ----------------------------
# Utility: KML parsing (extract polygons, points)
# ----------------------------
def parse_kml_coords(kml_path):
    """
    Parse a KML file and return a list of polygons. Each polygon is a list of (lat, lon) tuples.
    Handles <Polygon> and simple <Placemark><Point>.
    """
    ns = {"kml": "c"}
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # KML can nest Document/Folder/Placemark. Find all coordinate tags under polygons.
    polygons = []

    # find all <coordinates> text nodes
    for coord_elem in root.findall(".//{http://www.opengis.net/kml/2.2}coordinates"):
        text = coord_elem.text
        if not text:
            continue
        # coordinates are "lon,lat[,alt] ..." space or newline separated
        coords = []
        for part in text.strip().split():
            comps = part.strip().split(",")
            if len(comps) >= 2:
                lon = float(comps[0])
                lat = float(comps[1])
                coords.append((lat, lon))
        if coords:
            polygons.append(coords)

    # fallback: if nothing found, try a simpler search
    if not polygons:
        for pm in root.findall(".//Placemark"):
            for pt in pm.findall(".//{http://www.opengis.net/kml/2.2}Point"):
                coords_text = pt.find("{http://www.opengis.net/kml/2.2}coordinates")
                if coords_text is not None and coords_text.text:
                    comps = coords_text.text.strip().split(",")
                    lon, lat = float(comps[0]), float(comps[1])
                    polygons.append([(lat, lon)])

    return polygons


# ----------------------------
# Geo math and tile/grid helpers
# ----------------------------
def meters_per_pixel(lat_deg, zoom, scale):
    """Google Web Mercator ground resolution (m/px) adjusted by scale."""
    lat_rad = math.radians(lat_deg)
    return (156543.03392 * math.cos(lat_rad) / (2 ** zoom)) / scale


def tile_ground_extent_meters(lat_deg, zoom, scale, image_px=IMAGE_SIZE_PX):
    """Return side length in meters of a static tile (square)."""
    mpp = meters_per_pixel(lat_deg, zoom, scale)
    return image_px * mpp


def bbox_from_polygons(polygons):
    """Return (min_lat, min_lon, max_lat, max_lon) for a list of polygons [(lat,lon), ...]."""
    lats = []
    lons = []
    for poly in polygons:
        for lat, lon in poly:
            lats.append(lat)
            lons.append(lon)
    return min(lats), min(lons), max(lats), max(lons)


def degrees_per_tile_at_lat(lat_deg, zoom, scale):
    """Return (deg_lat_tile, deg_lon_tile) for a tile centered at lat_deg."""
    side_m = tile_ground_extent_meters(lat_deg, zoom, scale)
    # degrees: approximate conversion (valid for small extents)
    deg_per_m_lat = 1.0 / 111320.0
    deg_lat = side_m * deg_per_m_lat
    deg_lon = side_m * deg_per_m_lat / math.cos(math.radians(lat_deg))
    return deg_lat, deg_lon


def generate_tile_centers_for_bbox(min_lat, min_lon, max_lat, max_lon, zoom, scale):
    """
    Generate tile center coordinates (lat,lon) that cover the bbox.
    Uses the center latitude of the bbox for degree conversions but recalculates per-row for more accuracy.
    """
    centers = []

    # use mean lat for approximate tile size
    mean_lat = (min_lat + max_lat) / 2.0
    deg_lat_tile, deg_lon_tile = degrees_per_tile_at_lat(mean_lat, zoom, scale)

    # step sizes
    step_lat = deg_lat_tile
    step_lon = deg_lon_tile

    # compute starting center: start at min + half tile
    half_lat = step_lat / 2.0
    half_lon = step_lon / 2.0

    lat = min_lat + half_lat
    while lat <= max_lat - half_lat + 1e-12:
        # for each latitude row, recompute deg_lon_tile because cos(lat) changes
        _, deg_lon_tile_row = degrees_per_tile_at_lat(lat, zoom, scale)
        lon_step = deg_lon_tile_row
        lon = min_lon + (lon_step / 2.0)
        while lon <= max_lon - (lon_step / 2.0) + 1e-12:
            centers.append((round(lat, 12), round(lon, 12)))
            lon += lon_step
        lat += step_lat

    return centers


# ----------------------------
# Download Static Map tile
# ----------------------------
def download_static_map_tile(center_lat, center_lon, zoom, size_px, scale, api_key, maptype="satellite"):
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={center_lat},{center_lon}"
        f"&zoom={zoom}"
        f"&size={size_px}x{size_px}"
        f"&scale={scale}"
        f"&maptype={maptype}"
        f"&key={api_key}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


# ----------------------------
# Convert polygon lat/lon to pixel polygon for a given tile center & resolution
# ----------------------------
def polygon_latlon_to_pixel_coords(polygon_latlon, center_lat, center_lon, zoom, scale, image_px):
    """
    Convert polygon (list of (lat,lon)) to pixel coordinates in [0,image_px).
    Pixel origin is top-left; center corresponds to (image_px/2, image_px/2).
    """
    mpp = meters_per_pixel(center_lat, zoom, scale)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(center_lat))

    pixel_coords = []
    for lat, lon in polygon_latlon:
        dy_m = (lat - center_lat) * meters_per_deg_lat   # north positive
        dx_m = (lon - center_lon) * meters_per_deg_lon   # east positive

        px = (image_px / 2.0) + (dx_m / mpp)
        py = (image_px / 2.0) - (dy_m / mpp)  # decrease y when lat increases
        pixel_coords.append((px, py))
    return pixel_coords


# ----------------------------
# Model and prediction helpers (adapted from previous script)
# ----------------------------
def load_model_for_type(model_type, model_name, input_shape):
    if model_type == 1:
        model = fast_scnn_2.fast_scnn_v2(input_shape=input_shape, batch_size=1, n_labels=2, model_summary=False)
    elif model_type == 2:
        model = segnet_3.segnet_resnet_v2(input_shape=input_shape, batch_size=1, n_labels=2, model_summary=False)
    elif model_type == 3:
        model = segnet_1.segnet_4_encoder_decoder(input_shape=input_shape, batch_size=1, n_labels=2, model_summary=False)
    elif model_type == 4:
        model = segnet_0.segnet_original(input_shape=input_shape, batch_size=1, n_labels=2, model_summary=False)
    else:
        raise ValueError("Invalid model type")
    model.load_weights(os.path.join(MODEL_PATH, model_name))
    return model


def get_predicted_label_list(sub_imgs, model, threshold=CONF_THRESHOLD):
    sub_predicted_label_list = []
    total_batches = math.ceil(sub_imgs.shape[0] / INITIAL_BATCH_SIZE)
    for i in range(total_batches):
        start_idx = i * INITIAL_BATCH_SIZE
        if sub_imgs.shape[0] % INITIAL_BATCH_SIZE != 0 and i == total_batches - 1:
            batch_size = sub_imgs.shape[0] % INITIAL_BATCH_SIZE
        else:
            batch_size = INITIAL_BATCH_SIZE

        results = model.predict(sub_imgs[start_idx: start_idx + batch_size])
        for j in range(results.shape[0]):
            prob_map = results[j]
            custom_result = np.zeros_like(prob_map)
            object_mask = prob_map[:, :, 1] > threshold
            custom_result[object_mask, 1] = 1
            custom_result[~object_mask, 0] = 1
            my_img = prepare_data.onehot_to_rgb(custom_result, prepare_data.id2code)
            sub_predicted_label_list.append(my_img)
    return sub_predicted_label_list


# ----------------------------
# Processing a single tile: run model, clip to ROI, save outputs
# ----------------------------
def process_tile_image_and_save(tile_image_pil, center_lat, center_lon, zoom, scale, polygons_latlon,
                                model, saving_path, panel_area_m2=PANEL_AREA_M2, threshold=CONF_THRESHOLD):
    """
    tile_image_pil: PIL RGB image (1024x1024)
    polygons_latlon: list of polygons (each is list of (lat,lon)) -- ROI polygons from KML
    model: loaded model
    """
    # Save raw downloaded tile first (filename format)
    base_name = f"kmz-zm-{zoom}_{center_lat}_{center_lon}"
    os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
    tile_path = os.path.join(INPUT_IMAGE_DIR, base_name + ".png")
    tile_image_pil.save(tile_path, "PNG")

    # prepare image array
    image = np.asarray(tile_image_pil)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    original_width = image.shape[1]
    original_height = image.shape[0]

    # Sub-images using your helper (must return sub_imgs, padded_img, padded_width, padded_height)
    sub_imgs, padded_img, padded_width, padded_height = data_processing_tool_4.get_sub_images(image)

    # Run predictions using the model (sub tiles -> stitched label)
    sub_predicted_label_list = get_predicted_label_list(sub_imgs, model, threshold=threshold)
    full_label = data_processing_tool_4.get_full_predicted_label(padded_height, padded_width, sub_predicted_label_list)

    # convert to numpy array
    if isinstance(full_label, Image.Image):
        full_label_arr = np.array(full_label)
    else:
        full_label_arr = np.asarray(full_label)

    # compute predicted mask (RGB or single channel)
    if full_label_arr.ndim == 2:
        pred_mask = full_label_arr > 0
    elif full_label_arr.ndim == 3:
        if full_label_arr.shape[2] == 4:
            rgb_label = full_label_arr[:, :, :3]
        else:
            rgb_label = full_label_arr
        pixels = rgb_label.reshape(-1, rgb_label.shape[2])
        uniq_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        bg_color = uniq_colors[counts.argmax()]
        pred_mask = np.any(rgb_label != bg_color.reshape(1, 1, 3), axis=2)
    else:
        pred_mask = np.zeros((full_label_arr.shape[0], full_label_arr.shape[1]), dtype=bool)

    # Build ROI mask for the tile: union of all polygons projected into pixel coords
    roi_mask = Image.new("L", (IMAGE_SIZE_PX, IMAGE_SIZE_PX), 0)
    draw = ImageDraw.Draw(roi_mask)
    for poly in polygons_latlon:
        pix_poly = polygon_latlon_to_pixel_coords(poly, center_lat, center_lon, zoom, scale, IMAGE_SIZE_PX)
        # Clip polygon points to image bounds to avoid drawing error
        pix_poly_clipped = [(max(0, min(IMAGE_SIZE_PX - 1, x)), max(0, min(IMAGE_SIZE_PX - 1, y))) for x, y in pix_poly]
        # If polygon entirely outside tile, it will be drawn outside - drawing is safe
        draw.polygon(pix_poly_clipped, outline=1, fill=1)
    roi_mask_arr = np.array(roi_mask).astype(bool)

    # Intersect predicted mask with ROI mask
    # Ensure arrays sizes match (pred_mask may be padded -> crop to IMAGE_SIZE_PX)
    # full_label_arr shape may match padded image sizes. We must ensure roi_mask aligns with image size.
    # We assumed tile images are IMAGE_SIZE_PX x IMAGE_SIZE_PX. If full_label_arr differs, resize ROI appropriately.
    label_h, label_w = pred_mask.shape
    if (label_h, label_w) != (IMAGE_SIZE_PX, IMAGE_SIZE_PX):
        # Resize roi_mask_arr to label size using simple nearest neighbor scaling
        roi_pil_resized = Image.fromarray((roi_mask_arr.astype(np.uint8) * 255)).resize((label_w, label_h), resample=Image.NEAREST)
        roi_mask_arr = np.array(roi_pil_resized).astype(bool)

    masked_pred = pred_mask & roi_mask_arr

    mask_pixels = int(masked_pred.sum())
    total_pixels = masked_pred.shape[0] * masked_pred.shape[1]
    mask_percent = 100.0 * mask_pixels / (total_pixels if total_pixels > 0 else 1.0)

    # compute meters per pixel & mask area
    mpp = meters_per_pixel(center_lat, zoom, scale)
    pixel_area_m2 = mpp * mpp
    mask_area_m2 = mask_pixels * pixel_area_m2

    # estimate panels
    est_panels_float = mask_area_m2 / panel_area_m2 if panel_area_m2 > 0 else 0.0
    est_panels_int = int(math.floor(est_panels_float))

    # compute tile top-left & bottom-right lat/lon for metadata (based on center)
    half_m = (IMAGE_SIZE_PX / 2.0) * mpp
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(center_lat))
    dlat = half_m / meters_per_deg_lat
    dlon = half_m / meters_per_deg_lon
    top_left = (center_lat + dlat, center_lon - dlon)
    bottom_right = (center_lat - dlat, center_lon + dlon)

    # Save per-tile text file and overlayed image
    os.makedirs(saving_path, exist_ok=True)
    base_out = f"kmz-zm-{zoom}_{center_lat}_{center_lon}"
    txt_path = os.path.join(saving_path, base_out + ".txt")
    png_path = os.path.join(saving_path, base_out + ".png")

    # Save overlayed image using your helper
    full_label_with_mask = data_processing_tool_4.add_transparent_mask(padded_img, full_label, original_width, original_height)
    full_label_with_mask.save(png_path, "PNG")

    # Write text file
    with open(txt_path, "w") as fh:
        fh.write(f"input_tile: {base_out}.png\n")
        fh.write(f"center_lat: {center_lat}\n")
        fh.write(f"center_lon: {center_lon}\n")
        fh.write(f"top_left_lat: {top_left[0]}\n")
        fh.write(f"top_left_lon: {top_left[1]}\n")
        fh.write(f"bottom_right_lat: {bottom_right[0]}\n")
        fh.write(f"bottom_right_lon: {bottom_right[1]}\n")
        fh.write(f"zoom: {zoom}\n")
        fh.write(f"meters_per_pixel: {mpp:.6f}\n")
        fh.write(f"mask_pixels (within ROI): {mask_pixels}\n")
        fh.write(f"total_pixels (tile): {total_pixels}\n")
        fh.write(f"mask_percent: {mask_percent:.6f}%\n")
        fh.write(f"mask_area_m2: {mask_area_m2:.4f}\n")
        fh.write(f"panel_area_m2 (input): {panel_area_m2:.4f}\n")
        fh.write(f"estimated_panels_float: {est_panels_float:.6f}\n")
        fh.write(f"estimated_panels_int_floor: {est_panels_int}\n")
        fh.write(f"calc_timestamp_utc: {datetime.datetime.utcnow().isoformat()}Z\n")

    # Append to summary
    summary_file = os.path.join(saving_path, "areas_summary.txt")
    with open(summary_file, "a") as sf:
        sf.write(f"{base_out}\t{mask_pixels}\t{mask_percent:.6f}%\t{mask_area_m2:.4f}\t{est_panels_int}\n")

    print(f"Saved: {os.path.basename(png_path)} and {os.path.basename(txt_path)}  (mask_px={mask_pixels}, panels≈{est_panels_int})")

    return mask_pixels


# ----------------------------
# Orchestration: read kml, tile, download, process
# ----------------------------
def main():
    # 1) read polygons from KML
    polygons = parse_kml_coords(KML_PATH)
    if not polygons:
        raise RuntimeError("No coordinates found in KML - please check the file.")

    # unify polygons (we will pass all polygons to each tile and draw union)
    # but also compute bbox
    min_lat, min_lon, max_lat, max_lon = bbox_from_polygons(polygons)
    print(f"KML bbox: lat [{min_lat}, {max_lat}] lon [{min_lon}, {max_lon}]")

    # 2) generate tile centers
    centers = generate_tile_centers_for_bbox(min_lat, min_lon, max_lat, max_lon, GMAPS_ZOOM, GMAPS_SCALE)
    print(f"Generated {len(centers)} tile centers that cover the ROI.")

    if not centers:
        print("No tiles required (empty). Exiting.")
        return

    # 3) ensure dirs exist
    os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(SAVING_PATH, exist_ok=True)

    # 4) prepare model by loading with sample input shape
    # we need to get input shape from a dummy sub-image. We'll download the first tile temporarily
    first_lat, first_lon = centers[0]
    print("Downloading first tile to infer model input shape...")
    first_tile = download_static_map_tile(first_lat, first_lon, GMAPS_ZOOM, IMAGE_SIZE_PX, GMAPS_SCALE, API_KEY, GMAPS_MAPTYPE)
    first_tile_arr = np.asarray(first_tile)
    if first_tile_arr.ndim == 3 and first_tile_arr.shape[2] == 4:
        first_tile_arr = first_tile_arr[:, :, :3]

    sub_imgs_dummy, padded_img_dummy, pw_dummy, ph_dummy = data_processing_tool_4.get_sub_images(first_tile_arr)
    model = load_model_for_type(MODEL_TYPE, MODEL_NAME, input_shape=sub_imgs_dummy[0].shape)
    print("Model loaded.")

    # 5) Process each tile: download, process, and save. If ROI does not intersect tile, skip processing.
    for idx, (lat, lon) in enumerate(centers, 1):
        print(f"[{idx}/{len(centers)}] Tile center: {lat}, {lon}")
        try:
            tile_img = download_static_map_tile(lat, lon, GMAPS_ZOOM, IMAGE_SIZE_PX, GMAPS_SCALE, API_KEY, GMAPS_MAPTYPE)
        except Exception as e:
            print(f"Failed to download tile {lat},{lon}: {e}")
            continue

        # quick test: if polygon does not intersect tile bbox, optionally skip
        # compute tile bbox degrees and test polygon bbox intersection
        half_m = (IMAGE_SIZE_PX / 2.0) * meters_per_pixel(lat, GMAPS_ZOOM, GMAPS_SCALE)
        deg_lat_half = half_m / 111320.0
        deg_lon_half = half_m / (111320.0 * math.cos(math.radians(lat)))
        tile_min_lat = lat - deg_lat_half
        tile_max_lat = lat + deg_lat_half
        tile_min_lon = lon - deg_lon_half
        tile_max_lon = lon + deg_lon_half

        # quick polygon bbox check
        if max_lat < tile_min_lat or min_lat > tile_max_lat or max_lon < tile_min_lon or min_lon > tile_max_lon:
            print("Tile bbox does not intersect KML bbox; skipping heavy prediction.")
            # still save the tile image (optional). Here we skip processing.
            tile_base = f"kmz-zm-{GMAPS_ZOOM}_{lat}_{lon}"
            tile_img.save(os.path.join(INPUT_IMAGE_DIR, tile_base + ".png"))
            continue

        # Process tile and restrict counts to KML polygons
        try:
            process_tile_image_and_save(tile_img, lat, lon, GMAPS_ZOOM, GMAPS_SCALE, polygons, model, SAVING_PATH,
                                        panel_area_m2=PANEL_AREA_M2, threshold=CONF_THRESHOLD)
        except Exception as e:
            print(f"Error processing tile {lat},{lon}: {e}")

        # small pause between requests
        time.sleep(REQUEST_PAUSE_SECONDS)

    print("Done.")


if __name__ == "__main__":
    main()
