import cv2
import numpy as np
import gradio as gr
from PIL import Image
import mediapipe as mp
import matplotlib.pyplot as plt
# from deepface import DeepFace  # Disabled for performance

# Setup Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Detect face shape
def detect_face_shape(image, landmarks):
    jaw = [landmarks[i] for i in range(0, 17)]
    width = np.linalg.norm([jaw[0].x - jaw[-1].x, jaw[0].y - jaw[-1].y])
    height = np.linalg.norm([landmarks[8].x - ((jaw[0].x + jaw[-1].x) / 2), 
                             landmarks[8].y - ((jaw[0].y + jaw[-1].y) / 2)])
    ratio = width / height
    if ratio > 1.5:
        return "Round"
    elif ratio > 1.3:
        return "Oval"
    else:
        return "Long"

# Skin tone detection
def detect_skin_tone(image):
    face = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    avg = cv2.mean(face)[:3]
    l_value = avg[0]
    if l_value > 180:
        return "Fair"
    elif l_value > 130:
        return "Medium"
    else:
        return "Dark"

# Gender detection (DISABLED)
def detect_gender(image):
    return "Unknown"
    # try:
    #     result = DeepFace.analyze(image, actions=['gender'], enforce_detection=False)
    #     return result[0]['gender']
    # except:
    #     return "Unknown"

# Overlay makeup (optimized)
def overlay_makeup(image, color=(255, 0, 0), area='lips', landmarks=None):
    if landmarks is None:
        return image
    overlay = image.copy()
    alpha = 0.4
    points = []
    if area == 'lips':
        points = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17]
    elif area == 'blush':
        points = [234, 93, 132, 58]
    elif area == 'eyeliner':
        points = [33, 133, 160, 159, 158, 157, 173]
    img_h, img_w = image.shape[:2]
    mesh_points = [(int(landmarks[p].x * img_w), int(landmarks[p].y * img_h)) for p in points]
    cv2.fillPoly(overlay, [np.array(mesh_points, dtype=np.int32)], color)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# Product recommendation (dummy links)
def recommend_products(skin_tone):
    base_url = "https://example-affiliate.com/"
    products = {
        "Fair": "fair-tone-foundation",
        "Medium": "medium-tone-concealer",
        "Dark": "dark-tone-blush"
    }
    product_link = base_url + products.get(skin_tone, "universal-kit")
    return f"👉 [Recommended Product for {skin_tone} Skin]({product_link})"

# Final processing function
def process_image(uploaded_image):
    print("Image received")
    uploaded_image = uploaded_image.resize((300, 300))
    image = np.array(uploaded_image)

    try:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return image, "❌ No face detected in image."
        landmarks = results.multi_face_landmarks[0].landmark

        face_shape = detect_face_shape(image, landmarks)
        skin_tone = detect_skin_tone(image)
        gender = detect_gender(image)

        image = overlay_makeup(image, area='lips', landmarks=landmarks)
        image = overlay_makeup(image, area='blush', landmarks=landmarks)
        image = overlay_makeup(image, area='eyeliner', landmarks=landmarks)

        output_img = Image.fromarray(image)
        suggestions = f"""
        👩 Gender: {gender}
        💠 Face Shape: {face_shape}
        🎨 Skin Tone: {skin_tone}

        💄 Makeup Tips:
        - For {face_shape} face, contour to highlight cheekbones.
        - {skin_tone} skin tones look great with warm shades.
        - Try soft eyeliner and blush for a balanced look.

        {recommend_products(skin_tone)}
        """
        return output_img, suggestions.strip()
    except Exception as e:
        print(f"Processing error: {e}")
        return image, "❌ Error processing image."

# Gradio UI
gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Your Face Image"),
    outputs=[gr.Image(label="Makeup Applied"), gr.Textbox(label="Suggestions")],
    title="AI Makeup Recommender",
    description="Upload your photo to receive AI-based makeup suggestions and virtual try-on."
).launch(server_name="0.0.0.0", server_port=10000)
