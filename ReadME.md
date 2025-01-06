This is the face landmark recognition program.

- Source code
  main.py

- Build method

1. Change the mediapipe python module to change the model path for face landmark.
   mediapipe module path - C:\Users\Administrator\AppData\Roaming\Python\Python310\site-packages\mediapipe

2. Use this command - pyinstaller main.py
   pyinstaller is located at C:\Users\Administrator\AppData\Roaming\Python\Python310\Scripts

3. Then, copy the mediapipe module to the \_internal folder
   In the \_internal folder, you can see mediapipe folder and you must copy changed mediapipe module.
   This is because mediapipe folder contains some model files.

4. You can see the main.exe file on the dist folder.
