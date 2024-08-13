## README

### Overview
- This project includes two scripts for image processing using YOLO models: one for bounding box detection and another for segmentation tasks. Both scripts perform predictions on images from a specified folder and save the results to a database. The models used are selected based on CCTV camera IDs stored in a database, and the confidence values for predictions are also retrieved from the database.

### Prerequisites
- Python 3.x
- `ultralytics` library
- `pyodbc` library
- Other dependencies (install via pip)

### Inputs
- **Input Folder:** C:\input (Folder containing images to be processed)
- **Model Folder:** C:\model_pt (Folder containing YOLO model files)

### Parameters
- **save_to_project:** Directory to save prediction results.
- **name_new_folder:** Subdirectory name for saving images.
- **v_save:** Boolean flag to save prediction images.
- **v_conf:** Confidence threshold for predictions (default: 0.70, retrieved from database).
- **v_iou:** IoU threshold for non-maximum suppression (default: 0.45).
- **v_exite:** Boolean flag for saving all predictions in one folder.
- **image_size:** Size of the images for YOLO (640x640).

### Scripts

#### 1. Bounding Box Detection
- **Description:** Processes images to detect bounding boxes using YOLO models and saves the results to a database.
- **Functions:**
  - **read_images(input_folder):** Continuously reads images from the input folder and processes them.
  - **choose_model(cctv_id, image_path, source_name):** Finds the appropriate YOLO model name and confidence from the database using the CCTV ID and retrieves the corresponding model file from the model folder.
  - **find_model_file(model_param_code, model_folder):** Searches for the YOLO model file in the specified model folder based on the model parameter code retrieved from the database.
  - **predict_loop(model_ocr, cctv_id, v_conf):** Executes predictions using the selected model and parameters.
- **Class:**
  - **DetectionPredictorDB:** Custom YOLO predictor class that processes predictions and saves results to the database.

#### 2. Segmentation
- **Description:** Processes images to perform segmentation tasks using YOLO models and saves the results to a database.
- **Functions:**
  - **read_images(input_folder):** Continuously reads images from the input folder and processes them.
  - **choose_model(cctv_id, image_path, source_name):** Finds the appropriate YOLO model name and confidence from the database using the CCTV ID and retrieves the corresponding model file from the model folder.
  - **find_model_file(model_param_code, model_folder):** Searches for the YOLO model file in the specified model folder based on the model parameter code retrieved from the database.
  - **predict_loop(model_ocr, cctv_id, v_conf):** Executes predictions using the selected model and parameters.
- **Class:**
  - **SegmentationPredictorDB:** Custom YOLO predictor class for segmentation that processes predictions, applies masks, and saves results to the database.

### Example Usage
- To start processing images with bounding box detection, run:
  ```bash
  python Project_Detection.py
- To start processing images with segmentation, run:
   ```bash
  python Project_Segmentaion.py

<br>
<img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320">

# 📚 Ultralytics Docs

[Ultralytics](https://ultralytics.com) Docs are the gateway to understanding and utilizing our cutting-edge machine learning tools. These documents are deployed to [https://docs.ultralytics.com](https://docs.ultralytics.com) for your convenience.

[![pages-build-deployment](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment)
[![Check Broken links](https://github.com/ultralytics/docs/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/links.yml)
[![Check Domains](https://github.com/ultralytics/docs/actions/workflows/check_domains.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/check_domains.yml)
[![Ultralytics Actions](https://github.com/ultralytics/docs/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/format.yml)
[![Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://ultralytics.com/discord)

## 🛠️ Installation

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
[![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

To install the ultralytics package in developer mode, ensure you have Git and Python 3 installed on your system. Then, follow these steps:

1. Clone the ultralytics repository to your local machine using Git:

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    ```

2. Navigate to the cloned repository's root directory:

    ```bash
    cd ultralytics
    ```

3. Install the package in developer mode using pip (or pip3 for Python 3):

    ```bash
    pip install -e '.[dev]'
    ```

- This command installs the ultralytics package along with all development dependencies, allowing you to modify the package code and have the changes immediately reflected in your Python environment.

## 🚀 Building and Serving Locally

The `mkdocs serve` command builds and serves a local version of your MkDocs documentation, ideal for development and testing:

```bash
mkdocs serve
```

- #### Command Breakdown:

    - `mkdocs` is the main MkDocs command-line interface.
    - `serve` is the subcommand to build and locally serve your documentation.

- 🧐 Note:

    - Grasp changes to the docs in real-time as `mkdocs serve` supports live reloading.
    - To stop the local server, press `CTRL+C`.

## 🌍 Building and Serving Multi-Language

Supporting multi-language documentation? Follow these steps:

1. Stage all new language \*.md files with Git:

    ```bash
    git add docs/**/*.md -f
    ```

2. Build all languages to the `/site` folder, ensuring relevant root-level files are present:

    ```bash
    # Clear existing /site directory
    rm -rf site

    # Loop through each language config file and build
    mkdocs build -f docs/mkdocs.yml
    for file in docs/mkdocs_*.yml; do
      echo "Building MkDocs site with $file"
      mkdocs build -f "$file"
    done
    ```

3. To preview your site, initiate a simple HTTP server:

    ```bash
    cd site
    python -m http.server
    # Open in your preferred browser
    ```

- 🖥️ Access the live site at `http://localhost:8000`.

## 📤 Deploying Your Documentation Site

Choose a hosting provider and deployment method for your MkDocs documentation:

- Configure `mkdocs.yml` with deployment settings.
- Use `mkdocs deploy` to build and deploy your site.

* ### GitHub Pages Deployment Example:

    ```bash
    mkdocs gh-deploy
    ```

- Update the "Custom domain" in your repository's settings for a personalized URL.

![196814117-fc16e711-d2be-4722-9536-b7c6d78fd167](https://user-images.githubusercontent.com/26833433/210150206-9e86dcd7-10af-43e4-9eb2-9518b3799eac.png)

- For detailed deployment guidance, consult the [MkDocs documentation](https://www.mkdocs.org/user-guide/deploying-your-docs/).

## 💡 Contribute

We cherish the community's input as it drives Ultralytics open-source initiatives. Dive into the [Contributing Guide](https://docs.ultralytics.com/help/contributing) and share your thoughts via our [Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A heartfelt thank you 🙏 to each contributor!

![Ultralytics open-source contributors](https://github.com/ultralytics/assets/raw/main/im/image-contributors.png)

## 📜 License

Ultralytics Docs presents two licensing options:

- **AGPL-3.0 License**: Perfect for academia and open collaboration. Details are in the [LICENSE](https://github.com/ultralytics/docs/blob/main/LICENSE) file.
- **Enterprise License**: Tailored for commercial usage, offering a seamless blend of Ultralytics technology in your products. Learn more at [Ultralytics Licensing](https://ultralytics.com/license).

## ✉️ Contact

For bug reports and feature requests, navigate to [GitHub Issues](https://github.com/ultralytics/docs/issues). Engage with peers and the Ultralytics team on [Discord](https://ultralytics.com/discord) for enriching conversations!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="Ultralytics Instagram"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
