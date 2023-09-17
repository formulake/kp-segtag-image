import gradio as gr
import os
import cv2
import mediapipe as mp

print("Starting script...")

# Set the GRADIO_SERVER_PORT environment variable
os.environ["GRADIO_SERVER_PORT"] = "7850"

shot_types = ["upper_body", "cowboy_shot", "close_up", "portrait", "full_body", "unknown"]


def detect_damaged_eyes(image_path, blur_threshold):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    value = 0  # Initialize the value variable here

    for (ex, ey, ew, eh) in eyes:
        roi_gray = gray[ey:ey+eh, ex:ex+ew]
        value = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
        if value < blur_threshold:  # Use the dynamic threshold
            return True
    return False

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

def detect_face(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    if results.detections:
        # Calculate face size as a fraction of total image area
        h, w, _ = img.shape
        face_h = results.detections[0].location_data.relative_bounding_box.height * h
        face_w = results.detections[0].location_data.relative_bounding_box.width * w
        face_size = face_h * face_w / (h * w)
        return True, face_size
    return False, 0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def pose_estimation(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        keypoints = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[f"landmark_{idx}"] = (landmark.x, landmark.y, landmark.visibility)  # Include visibility
        return keypoints
    return None


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_body_with_holistic(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    
    if results.pose_landmarks:
        return True
    return False

def classify_shot_type(img_path):
    img = cv2.imread(img_path)

    # Detect face and its size
    has_face, face_size = detect_face(img)

    # Close-Up: Only the face or head and neck are visible
    if has_face and face_size > 0.5:  # Adjusted threshold
        return "close_up"

    # Portrait: Only head and chest
    if has_face and face_size > 0.2:  # Adjusted threshold
        return "portrait"

    # Estimate pose keypoints
    keypoints = pose_estimation(img)

    # If no keypoints detected, default to "unknown"
    if not keypoints:
        return "unknown"

    # Check visibility of keypoints
    visibility_threshold = 0.8  # You can adjust this threshold
    keypoints_visible = {k: v for k, v in keypoints.items() if v[2] > visibility_threshold}  # v[2] is the visibility

    # Check for keypoints
    nose = keypoints_visible.get("landmark_0", None)
    left_hip = keypoints_visible.get("landmark_11", None)
    right_hip = keypoints_visible.get("landmark_12", None)
    left_ankle = keypoints_visible.get("landmark_15", None)
    right_ankle = keypoints_visible.get("landmark_16", None)
    left_shoulder = keypoints_visible.get("landmark_5", None)
    right_shoulder = keypoints_visible.get("landmark_6", None)

    # Cowboy Shot: From thighs to head, not including feet or ankles
    if nose and left_hip and right_hip and not left_ankle and not right_ankle:
        return "cowboy_shot"

    # Upper Body: From head to hips or waist
    if nose and left_hip and right_hip and left_shoulder and right_shoulder and not left_ankle and not right_ankle:
        return "upper_body"

    # Full Body: All major keypoints from head to ankles are detected
    if nose and left_hip and right_hip and left_shoulder and right_shoulder and left_ankle and right_ankle:
        return "full_body"

    # Default to "unknown" if no other classifications match
    return "unknown"



def segregate_images(source_folder, upper_body, cowboy, close_up, portrait, full_body, blur_threshold):
    # Create necessary directories within the source folder
    folders = ["damaged_eyes"] + shot_types
    for folder in folders:
        if not os.path.exists(os.path.join(source_folder, folder)):
            os.mkdir(os.path.join(source_folder, folder))

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                if detect_damaged_eyes(img_path, blur_threshold):
                    os.rename(img_path, os.path.join(source_folder, "damaged_eyes", file))
                else:
                    classification = classify_shot_type(img_path)
                    os.rename(img_path, os.path.join(source_folder, classification, file))
    
    return f"Images from {source_folder} have been processed."

holistic.close()

# Define the functions for the blocks
def combined_functions(source_folder, upper_body, cowboy, close_up, portrait, full_body, blur_threshold, 
                       hairstyle_input, clothing_input, wd14_input):
    angle_output = segregate_images(source_folder, upper_body, cowboy, close_up, portrait, full_body, blur_threshold)
    hairstyle_output = hairstyle_segregator(hairstyle_input)
    clothing_output = clothing_segregator(clothing_input)
    wd14_output = wd14_tagging(wd14_input)
    return angle_output, hairstyle_output, clothing_output, wd14_output

# Create the interface
interface = gr.Interface(
    fn=combined_functions,
    inputs=[
        gr.Textbox(label="Angle Segregator: Provide the path to your source folder here. All images to be segregated need to be present in the source folder. Subfolders will not be scanned."),
        gr.Checkbox(label="Upper Body"),
        gr.Checkbox(label="Cowboy"),
        gr.Checkbox(label="Close-Up"),
        gr.Checkbox(label="Portrait"),
        gr.Checkbox(label="Full Body"),
        gr.Slider(minimum=0, maximum=100, label="Blur Threshold for Eyes"),
        gr.Textbox(label="Placeholder input for Hairstyle Segregator"),
        gr.Textbox(label="Placeholder input for Clothing Segregator"),
        gr.Textbox(label="Placeholder input for WD14 Tagging")
    ],
    outputs=[
        gr.Textbox(label="Angle Segregator Output"),
        gr.Textbox(label="Hairstyle Segregator Output"),
        gr.Textbox(label="Clothing Segregator Output"),
        gr.Textbox(label="WD14 Tagging Output")
    ],
    live=False
)

# Launch the interface
interface.launch(server_port=7850)

# This script will group the inputs and outputs together in a single page. You can adjust the layout and appearance as needed.

