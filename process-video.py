import cv2
import mediapipe as mp
from google.cloud import storage

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# Define the indices of the lips in the face mesh
LIPS_INDICES = list(range(13, 16)) + list(range(78, 94)) + list(range(185, 201)) + list(range(310, 326))

def detect_lip_movement(frame, prev_lip_position, threshold=10):
    try:
        print("Processing frame for lip movement detection...")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                lip_positions = []
                for idx in LIPS_INDICES:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * iw)
                    y = int(landmark.y * ih)
                    lip_positions.append((x, y))
                
                # Calculate the centroid of the lip positions
                avg_x = int(sum([pos[0] for pos in lip_positions]) / len(lip_positions))
                avg_y = int(sum([pos[1] for pos in lip_positions]) / len(lip_positions))
                current_lip_position = (avg_x, avg_y)
                
                if prev_lip_position:
                    # Calculate the distance between the current and previous lip positions
                    distance = ((current_lip_position[0] - prev_lip_position[0])**2 +
                                (current_lip_position[1] - prev_lip_position[1])**2)**0.5
                    if distance > threshold:
                        print("Lip movement detected!")
                        return True, current_lip_position
                
                return False, current_lip_position
        
        print("No face landmarks detected in this frame.")
        return False, None
    except Exception as e:
        print(f"Error in detect_lip_movement: {e}")
        return False, None

def download_from_gcs(gcs_path, local_path):
    try:
        print(f"Downloading file from GCS: {gcs_path} to {local_path}")
        client = storage.Client()
        bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        print(f"File downloaded successfully: {local_path}")
    except Exception as e:
        print(f"Error downloading file from GCS: {e}")

def upload_to_gcs(local_path, gcs_path):
    try:
        print(f"Uploading file to GCS: {local_path} to {gcs_path}")
        client = storage.Client()
        bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        print(f"File uploaded successfully: {gcs_path}")
    except Exception as e:
        print(f"Error uploading file to GCS: {e}")

def process_video(input_path, output_path):
    try:
        print("Starting video processing pipeline...")
        
        # Step 1: Download the input video from GCS
        local_input_path = "/tmp/input_video.mp4"
        download_from_gcs(input_path, local_input_path)
        
        # Step 2: Process the video locally
        print("Opening video file for processing...")
        cap = cv2.VideoCapture(local_input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {local_input_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video properties: FPS={fps}, Width={width}, Height={height}")
        
        local_output_path = "/tmp/output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(local_output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Could not create output video file: {local_output_path}")
            return
        
        prev_lip_position = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break
            
            lip_moving, current_lip_position = detect_lip_movement(frame, prev_lip_position)
            prev_lip_position = current_lip_position
            out.write(frame)
        
        cap.release()
        out.release()
        print("Video processing completed successfully.")
        
        # Step 3: Upload the output video to GCS
        upload_to_gcs(local_output_path, output_path)
    
    except Exception as e:
        print(f"Error in process_video: {e}")

# Example usage
if __name__ == "__main__":
    input_path = "gs://v-input/for-cropping-jerry.mp4"
    output_path = "gs://v-input/cropped-jerry.mp4"
    process_video(input_path, output_path)