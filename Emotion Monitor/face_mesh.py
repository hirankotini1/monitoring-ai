import cv2
import mediapipe as mp

# Open camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not open camera.")
    exit()

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Tell MediaPipe where the model is
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# Create face landmarker
with FaceLandmarker.create_from_options(options) as landmarker:

    frame_number = 0

    while True:

        # Get frame from webcam
        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        # OpenCV uses BGR
        # MediaPipe uses RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp
        timestamp = frame_number * 33
        frame_number += 1

        # Detect facial landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        # If a face was found
        if result.face_landmarks:

            face = result.face_landmarks[0]

            height, width, _ = frame.shape

            # Draw all facial landmarks
            for landmark in face:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

        # Show camera
        cv2.imshow("Face Mesh", frame)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

camera.release()
cv2.destroyAllWindows()