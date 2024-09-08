import pydicom as pydicom
import utils.dicom_utils as dicom_utils
import model_20210820_XNet38MS.predict as predict
import gradio as gr
import glob
from PIL import Image
import numpy as np


model = predict.build_model()

def inference_image(filename):
    image_original = None
    display_image = None

    if filename.endswith('png'):
        display_image = Image.open(filename).convert("L")
        image_original = np.array(display_image)

    else:
        dicom = pydicom.read_file(filename)
        image_original = dicom_utils.img_clean(dicom)
        display_image = Image.fromarray((image_original).astype(np.uint8))
        
    report = predict.main(image_original, model)
    
    return display_image, report['AI_urgency'], report['AI_prediction']

demo = gr.Interface(
    fn=inference_image,
    inputs=gr.File(type="filepath"),
    outputs=[gr.Image(), gr.Label(), gr.Label(num_top_classes=38)],
    examples=glob.glob("../demo_data/*"),
)

demo.launch(share=True)

