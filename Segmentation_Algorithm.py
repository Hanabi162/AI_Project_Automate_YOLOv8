from ultralytics import YOLO
import os
import time
import pyodbc
from collections import Counter
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils import DEFAULT_CFG, ops
# I don't want to show the database connection details in this script on GitHub even if it is called through another file.

# Input
input_folder = r"your_Path" # A folder to store all your photos.
model_folder = r"your_Path" # Folder containing all models

# Parameters
save_to_project = r'your_Path' # Folder path to save
name_new_folder = r'your_Name' # Name of the folder where the desired results are stored.
v_save = True
v_iou = 0.7
v_exite = True
image_size = (640,640)

# Function for reading infinite loop format values ​​from a folder.
def read_images(input_folder):
    while True:
        if os.path.exists(input_folder):
            files = os.listdir(input_folder)
            time.sleep(2)
            if files:
                for file in files:
                    try:
                        image_path = os.path.join(input_folder, file)
                        source_name = os.path.basename(image_path)
                        cctv_id = os.path.basename(file)[:11]
                        choose_model(cctv_id, image_path, source_name)
                    except Exception as e:
                        print(f"Missing Image: {e}")
                print("\n All detections in the set are complete, waiting for the next set of images. \n")
            else:
                print("The folder is empty.")
                time.sleep(2)
        else:
            print("This folder doesn't actually exist.")
            break

# Function to select model from database, refer to CCTV, predict result and delete image.
def choose_model(cctv_id, image_path, source_name):
    try:
        cnxn = pyodbc.connect('Do Not Show Connection Details')
        cursor = cnxn.cursor()

        # This is a rewritten command for example only.
        query = """SELECT s.[conf], m.[model] 
        ... 
        """ # Undisclosed connections
        cursor.execute(query, cctv_id)
        model_param = cursor.fetchone()

        if model_param:
            v_conf = model_param[0]
            model_param_code = model_param[1]
            model_ocr = find_model_file(model_param_code, model_folder)
            print(f"CCTV-ID: {cctv_id} Model Requirements (from Database): {model_param_code}")
            print(f"Model Actually (from files): {model_ocr}")
            predict_loop(model_ocr, image_path, cctv_id, source_name,v_conf)
            os.remove(image_path)
            print("Predicting success and deleting predictions")

        else:
            print(f"No model parameter code found for CCTV ID {cctv_id}")
            model_param_code = None
            os.remove(image_path)
            print("Delete images that do not have a model in the database")

        cursor.close()
        cnxn.close()
    except Exception as e:
        print(f"Error accessing database or the model file was not found in the folder : {e}")
        os.remove(image_path)
        print("Remove unpredictable images")

# Function to find models whose names match the values ​​retrieved from the database.
def find_model_file(model_param_code, model_folder):
    for root, dirs, files in os.walk(model_folder):
        for file in files:
            if file.startswith(model_param_code):
                return os.path.join(root, file)
    return None

# Class for predicting images and performing string concatenation to send to the database.
class SegmentationPredictorDB(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, cctv_id=None, source_name=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"
        self.cctv_id = cctv_id
        self.source_name = source_name

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        cnxn = pyodbc.connect('Do Not Show Connection Details')
        cursor = cnxn.cursor()
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                # Example of retrieving all classes and counts and values ​​in SQL format
                sqlstr = f"""
                INSERT INTO Example (
                    id, cctv_id, detects
                ) VALUES (
                    NEXT VALUE FOR id, '{self.cctv_id}', 0
                )
                """
            else:
                class_set_index = set()
                for clsi in pred[:, 5]: 
                    class_index = int(clsi)
                    if class_index not in class_set_index: 
                        class_set_index.add(class_index)
                classes_idx = [f"otrans_class_{idx:02}" if idx < 10 else f"class_{idx}" for idx in sorted(class_set_index)]
                class_indices = [int(clsn) for clsn in pred[:, 5]]
                class_counts = Counter(class_indices)
                counts_list = [class_counts[idx] for idx in sorted(class_set_index)]
                classes_idx_str = ', '.join(classes_idx)
                classes_num = ', '.join(map(str, counts_list))
                trans_detects_All = sum(counts_list)

                # Example of retrieving all classes and counts and values ​​in SQL format
                sqlstr = f"""
                INSERT INTO Example (
                    id, cctv_id, {classes_idx_str}, detects
                ) VALUES (
                    NEXT VALUE FOR id, '{self.cctv_id}', {classes_num}, {trans_detects_All}
                )
                """

            cursor.execute(sqlstr)
            cnxn.commit()

            if self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

# Function to call and pass parameters to the prediction class.
def predict_loop(model_ocr, cctv_id,v_conf):
    print(f"Confident : {v_conf}")
    args = dict(model=model_ocr, conf=v_conf, iou=v_iou, save=v_save, project=save_to_project, name=name_new_folder, exist_ok=v_exite, imgsz=image_size)
    predictor = SegmentationPredictorDB(overrides=args, cctv_id=cctv_id)
    predictor.predict_cli()

    # To disable the on-screen output, go to:
    # ultralytics => utils => __init__ 
    # => VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode 
    # to
    # => VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "false"  # global verbose mode

if __name__ == "__main__":
    read_images(input_folder)