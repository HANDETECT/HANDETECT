import gradio as gr
import predict as predict
from googletrans import Translator, constants
from pprint import pprint

translator = Translator()


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def process_file(webcam_filepath, upload_filepath):
    result = []
    if webcam_filepath == None:
        sorted_classes = predict.predict_image(upload_filepath)
        for class_label, class_prob in sorted_classes:
            class_prob = class_prob.item().__round__(2)
            result.append(f"{class_label}: {class_prob}%")
        return result
    elif upload_filepath == None:
        sorted_classes = predict.predict_image(webcam_filepath)
        for class_label, class_prob in sorted_classes:
            class_prob = class_prob.item().__round__(2)
            result.append(f"{class_label}: {class_prob}%")
        return result
    else:
        sorted_classes = predict.predict_image(upload_filepath)
        for class_label, class_prob in sorted_classes:
            class_prob = class_prob.item().__round__(2)
            result.append(f"{class_label}: {class_prob}%")
        return result


def generate_description(request: gr.Request):
    translation = translator.translate(
        "SqueezeNet-Based Deep Learning for Early Detection of Movement Disorders via Handwriting Assessment",
        dest=str(request.request.headers["Accept-Language"].split(",")[0].lower()[0:2]),
    )
    return translation.text


demo = gr.Interface(
    theme="gradio/soft",
    fn=process_file,
    title="HANDETECT",
    # description=generate_description,
    inputs=[
        gr.components.Image(type="filepath", label="Choose Image", source="upload"),
    ],
    outputs=[
        gr.outputs.Textbox(label="Probability 1"),
        gr.outputs.Textbox(label="Probability 2"),
        gr.outputs.Textbox(label="Probability 3"),
    ],
)

demo.launch(inbrowser=True)
