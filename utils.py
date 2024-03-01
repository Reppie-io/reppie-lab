import io
import base64

from PIL import Image

def base64_to_pil(base64_str):
    img_data = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(img_data))
    
    return pil_image