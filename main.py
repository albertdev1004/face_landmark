import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import random
import shutil

def open_image():
  global original_image, original_image_path  # Store the original image for resizing
  file_path = filedialog.askopenfilename(
    filetypes=[("All files", "*.*")]
  )
  if file_path:
    # Load the original image
    original_image_path = file_path
    original_image = Image.open(file_path)
    display_image(original_image)

def display_image(image):
  """Resize and display the image dynamically."""
  # Get dimensions from the sliders
  new_width = 500
  new_height = 500

  # Ensure the dimensions are valid integers
  if isinstance(new_width, int) and isinstance(new_height, int):
    # Resize the image to the specified dimensions
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(resized_image)

    # Update the label with the resized image
    image_label.config(image=img_tk)
    image_label.image = img_tk
  else:
    print("Error: width and height must be integers")

def update_image(event=None):
    """Update the image display when sliders are adjusted."""
    if original_image:
      display_image(original_image)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Function to read an image
def read_image(image_path):
  # Create a temporary path with ASCII characters
  temp_path = os.path.join("./temp_image.jpg")
      
  # Copy the file to the temporary path
  shutil.copy(image_path, temp_path)
  image = cv2.imread(temp_path)
  if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")
  return image

# Function to process face landmarks
def process_face_landmarks(image):
  with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
  ) as face_mesh:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
  return results

# Function to save face landmarks to a file
def save_landmarks_to_file(results, output_path):
  with open(output_path, 'w') as f:
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        for i, landmark in enumerate(face_landmarks.landmark):
          f.write(f"Landmark {i}: x={landmark.x:.6f}, y={landmark.y:.6f}, z={landmark.z:.6f}\n")
      print(f"Landmarks saved to {output_path}")
    else:
      print("No face landmarks detected.")

# Function to convert normalized coordinates to pixel coordinates
def normalized_to_pixel_coordinates(landmark, image_shape):
  h, w, _ = image_shape
  return int(landmark.x * w), int(landmark.y * h)

# Function to draw random numbered landmarks on the image
def draw_random_landmarks(image, results):
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      h, w, _ = image.shape

      # Generate unique random numbers for each landmark
      num_landmarks = len(face_landmarks.landmark)
      random_numbers = random.sample(range(1, num_landmarks + 1), num_landmarks)

      for i, landmark in enumerate(face_landmarks.landmark):
        x, y = normalized_to_pixel_coordinates(landmark, image.shape)
        cv2.putText(
          image,
          str(i),
          (x, y),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 0, 255),
          1
        )

# Function to draw custom points
CUSTOM_LANDMARKS = [50, 280, 214, 434]  # Example indices for cheeks and mouth

def draw_custom_points(image, results):
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      h, w, _ = image.shape

      custom_points = [
        normalized_to_pixel_coordinates(face_landmarks.landmark[idx], image.shape)
        for idx in CUSTOM_LANDMARKS
      ]

      for point in custom_points:
        cv2.circle(image, point, 8, (255, 0, 0), -1)

def draw_face_outline(image, results):
  """
  Draws the face outline based on the provided landmarks.

  Parameters:
  - image: The image on which to draw.
  - face_landmarks: The detected face landmarks object.
  - width: Width of the image (used to convert normalized coordinates to pixels).
  - height: Height of the image (used to convert normalized coordinates to pixels).
  """
  # Define the indices of the landmarks for the face outline
  outline_indices = [
    10, 109, 67, 103, 103, 54, 21, 162, 127, 234,
    93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
    356, 389, 251, 284, 332, 297, 338, 10
  ]

  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      h, w, _ = image.shape
      # Convert normalized coordinates to pixel values
      points = [
        (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
        for idx in outline_indices
      ]

      # Draw the outline on the image
      cv2.polylines(image, [np.array(points)], isClosed=False, color=(255, 255, 0), thickness=2)  # Yellow outline

      return image

def generate_output_filename(input_filename, output_str):
  # Split the input filename into base name and extension
  base_name, ext = os.path.splitext(input_filename)
  
  # Add '_output' to the base name
  output_filename = base_name + output_str + ext
  
  return output_filename

def save_image(image, save_path):
  try:
    # Create a temporary ASCII-compatible path
    temp_path = "./temp_image.jpg"
    
    # Save the image using OpenCV
    cv2.imwrite(temp_path, image)
    
    # Move the temporary file to the desired Unicode path
    shutil.move(temp_path, save_path)
    print(f"Image successfully saved to: {save_path}")
  except Exception as e:
    raise IOError(f"Failed to save image to {save_path}. Error: {e}")

def process_image():
  if not original_image_path:
    print("No image selected!")
    return
  # Read and process the image
  image = read_image(original_image_path)
  results = process_face_landmarks(image)
  
  image_backone = read_image(original_image_path)
  image_backtwo = read_image(original_image_path)
  image_outline = read_image(original_image_path)
  
  # Draw random numbered landmarks
  draw_random_landmarks(image_backone, results)
  
  output_str = '_output_num'
  output_path_allnumber = generate_output_filename(original_image_path, output_str)
  # cv2.imwrite(output_path_allnumber, image_backone)
  save_image(image_backone, output_path_allnumber)

  # Draw outline using landmark
  draw_face_outline(image_outline, results)
  
  output_str = '_output_out'
  output_path_outline = generate_output_filename(original_image_path, output_str)
  # cv2.imwrite(output_path_outline, image_outline)
  save_image(image_outline, output_path_outline)
  
  # Draw custom points
  draw_custom_points(image_backtwo, results)
  
  output_str = '_output_pot'
  output_path_point = generate_output_filename(original_image_path, output_str)
  # cv2.imwrite(output_path_point, image_backtwo)
  save_image(image_backtwo, output_path_point)
  
  # Draw random numbered landmarks
  draw_random_landmarks(image, results)

  # Draw custom points
  draw_custom_points(image, results)

  # Draw outline using landmark
  draw_face_outline(image, results)
  
  output_str = '_output_all'
  output_path = generate_output_filename(original_image_path, output_str)
  
  # cv2.imwrite(output_path, image)
  save_image(image, output_path)
  
  original_image = Image.open(output_path)
  display_image(original_image)

# Create a Tkinter window
root = tk.Tk()
root.title("顔認証プログラム")
root.geometry("800x600")

original_image = None # Store the original image for resizing
original_image_path = "" # Store the original image path for resizing

# Button to open an image
open_button = tk.Button(root, text="画像選択", command=open_image)
open_button.pack(pady=10)

# Button to process an image
open_button = tk.Button(root, text="分析", command=process_image)
open_button.pack(pady=10)

# Label to display the image
image_label = tk.Label(root, bg="gray", width=800, height=500)
image_label.pack(expand=True, fill=tk.BOTH)

# Run the Tkinter main loop
root.mainloop()