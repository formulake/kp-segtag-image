# kp-segtag-image
Image Segmentation and Tagging

SegTag
segtag.py is a script designed to segregate and classify images based on various criteria such as facial features, body posture, and more. It leverages computer vision techniques and the Gradio interface to provide an interactive GUI for users.

Features
Angle Segregator: Classifies images based on the angle of the shot (e.g., close-up, portrait, full body).
Hairstyle Segregator: Uses the Roboflow API to classify hairstyles in images (integration in progress).
Clothing Segregator: Placeholder function for future development.
WD14 Tagging: Placeholder function for future development.
Prerequisites
Python 3.x
Gradio v3.44.3
OpenCV
Mediapipe
Roboflow (for Hairstyle Segregator)
Installation
Clone the repository:

bash
Copy code
git clone <repository-url>
Navigate to the directory:

bash
Copy code
cd path-to-directory
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Run the script:

bash
Copy code
python segtag.py
Open the Gradio interface in your browser using the provided link.

Use the GUI to select the desired functionality and provide the necessary inputs.

Contributing
Contributions are welcome! Please create a pull request with your changes.

License
This project is licensed under the MIT License.
