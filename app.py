import webview
import sys
import os

from ensemble_inference import run_bounding_box_only, run_box_and_segmentation

# sys.path.append('../automatic-annotation-system')
# from inference import your_inference_function
from config import cancel_flag


class Api:

    def process_image(self, image_path, prompt, mode):
        cancel_flag["stop"] = False
        if mode == 'bbox':
            print("begin processing image")
            output_path = run_bounding_box_only(image_path, prompt)
        elif mode == 'bbox+seg':
            output_path = run_box_and_segmentation(image_path, prompt)
        else:
            raise ValueError("Unknown mode selected.")
        return output_path

    def cancel_processing(self):
        cancel_flag["stop"] = True
        print("🛑 Cancel requested from frontend.")
        return True

    def choose_folder(self):
        result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        if result:
            return result[0]
        else:
            return None

    def list_files(self, folder_path):
        image_exts = (".jpg", ".jpeg", ".png")
        return [f for f in os.listdir(folder_path) if f.lower().endswith(image_exts)]


if __name__ == '__main__':
    api = Api()
    webview.create_window('Automatic Annotation GUI', 'frontend/index.html', js_api=api, width=1350, height=900)
    webview.start(http_server=True, debug=True)
