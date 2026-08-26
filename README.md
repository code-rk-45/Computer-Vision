# Computer Vision

## Solar Panel Detection

This project uses satellite imagery and a trained deep-learning segmentation
model to detect solar panels within a geographic area. The workflow uses a KML
file to define the area of interest, downloads satellite-image tiles for that
area through the Google Static Maps API, and analyzes each tile for solar-panel
pixels.

The program produces an estimate of:

- The area covered by detected solar panels, in square meters.
- The number of solar panels, calculated from the detected area and the
  configured `PANEL_AREA_M2` value in `solar_kml.py`.

The panel count is an estimate. Its accuracy depends on the satellite-image
resolution, segmentation model, KML boundary, map imagery, and configured
average panel area.

## Project structure

The solar-panel project is located in `Solar-panel-detection/`:

```text
Solar-panel-detection/
├── kml/                         # KML files defining areas of interest
├── save_input/                  # Downloaded satellite tiles
├── save_output/                 # Detection images and reports
├── trained_models/              # Saved TensorFlow model weights
├── data_processing/             # Image tiling and result-processing helpers
├── model_list/                  # Available segmentation model definitions
├── requirements.txt             # Python dependencies
└── solar_kml.py                 # Main KML-to-detection script
```

## Setup

Use Python 3 and install the dependencies:

```bash
cd Solar-panel-detection
python3 -m pip install -r requirements.txt
```

Before running the program, open `solar_kml.py` and configure:

- `API_KEY`: a Google Maps API key with access to the Static Maps API.
- `KML_PATH`: the KML file containing the area to analyze. The default is
  `kml/lastest_all.kml`.
- `MODEL_PATH` and `MODEL_NAME`: the model weights to use. The default model is
  `trained_models/fast_scnn_2.h5`.
- `PANEL_AREA_M2`: the assumed area of one solar panel. The default is `2.57`
  square meters.

The KML coordinates should use the standard KML order of longitude, latitude,
and optional altitude. The script supports polygon coordinates and simple point
coordinates.

## Running the detection

Run the script from the `Solar-panel-detection` directory:

```bash
python3 solar_kml.py
```

The script calculates the KML bounding box, downloads satellite tiles at the
configured zoom and scale, runs the segmentation model, restricts detections to
the KML region, and saves the results.

## Outputs

Downloaded input tiles are saved in `save_input/` using names such as:

```text
kmz-zm-19_<latitude>_<longitude>.png
```

For each processed tile, `save_output/` contains:

- A PNG overlay showing the detected solar-panel mask.
- A TXT report containing coordinates, mask pixels, panel-covered area, and the
  estimated panel count.

The script also appends a combined `areas_summary.txt` file containing the
per-tile area and panel estimates.

## Attribution and usage policy

This project was created by **Rohin Kar**. Before using, copying, modifying,
publishing, or redistributing this project or its results, please provide clear
attribution to Rohin Kar and cite the project author’s page:

[https://github.com/code-rk-45](https://github.com/code-rk-45)

Unless a separate license file or written permission says otherwise, this
attribution requirement applies to both academic and commercial use. Google
Maps imagery and API usage are subject to Google’s terms, usage limits, and
attribution requirements; users are responsible for complying with those terms
and for securing any required permissions before using the imagery or results.
