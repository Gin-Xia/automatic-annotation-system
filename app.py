import webview
import sys
import os

# sys.path.append('../automatic-annotation-system')
# from inference import your_inference_function

class Api:
    def process_image(self, image_path, prompt, mode):
        if mode == 'bbox':
            output_path = run_bounding_box_only(image_path, prompt)
        elif mode == 'bbox+seg':
            output_path = run_box_and_segmentation(image_path, prompt)
        else:
            raise ValueError("Unknown mode selected.")
        return output_path

    def choose_folder(self):
        result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        if result:
            return result[0]
        else:
            return None



if __name__ == '__main__':
    api = Api()
    webview.create_window('Automatic Annotation GUI', 'frontend/index.html', js_api=api)
    webview.start()
