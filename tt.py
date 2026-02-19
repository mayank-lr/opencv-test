# # Load model directly
from transformers import AutoModelForImageSegmentation
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True, dtype="auto")


# from transformers import pipeline
# image_path = "/home/logicrays/Desktop/botpress/files/Gemini_Generated_Image_37qbuk37qbuk37qb.png"
# pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
# pillow_mask = pipe(image_path, return_mask = True) # outputs a pillow mask
# pillow_image = pipe(image_path) # applies mask on input and returns a pillow image
