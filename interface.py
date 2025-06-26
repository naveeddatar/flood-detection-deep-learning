import gradio as gr
from fastai.vision.all import *
import PIL

# Load the trained model
learn = load_learner("export.pkl")  # Make sure this file is in the same folder or provide the correct path

# Define prediction function
def predict(image):
    pred_class, pred_idx, probs = learn.predict(image)
    return {
        "No Flood": float(probs[0]),
        "Flood": float(probs[1])
    }

# Create the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Flood Detection from Satellite Image",
    description="Upload a satellite image to check for flood damage using a deep learning model."
)

# Launch the app
if __name__ == "__main__":
    demo.launch()