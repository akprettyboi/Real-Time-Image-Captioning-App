from transformers import BlipProcessor, TFBlipForConditionalGeneration
from PIL import Image
import tensorflow as tf

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
processor.tokenizer.padding_side = 'left'

model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("C:\\Users\\Ayush\\OneDrive - BBTech\\Pictures\\WhatsApp Image 2024-06-30 at 15.56.17_7b1ecdb8.jpg")  # Replace with an actual image path
inputs = processor(images=image, return_tensors="tf", padding=True)
pixel_values = inputs["pixel_values"]

outputs = model.generate(pixel_values, max_length=30, num_beams=4)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print("Generated Caption:", caption)
