import cv2
import mediapipe as mp
from google.cloud import storage
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# Define the indices of the lips in the face mesh
LIPS_INDICES = (
    list(range(13, 16))
    + list(range(78, 94))
    + list(range(185, 201))
    + list(range(310, 326))
)

def detect_lip_movement(frame, prev_lip_position, threshold=10):
    """
    Detect lip movement by computing the centroid of lip landmarks and
    measuring distance from the previous centroid.
    Returns (lip_moving, current_lip_position).
    """
    try:
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

                # Calculate centroid of the lip region
                avg_x = int(sum(pos[0] for pos in lip_positions) / len(lip_positions))
                avg_y = int(sum(pos[1] for pos in lip_positions) / len(lip_positions))
                current_lip_position = (avg_x, avg_y)

                if prev_lip_position:
                    distance = (
                        (current_lip_position[0] - prev_lip_position[0]) ** 2
                        + (current_lip_position[1] - prev_lip_position[1]) ** 2
                    ) ** 0.5
                    if distance > threshold:
                        # Lip movement detected
                        return True, current_lip_position

                return False, current_lip_position

        return False, None

    except Exception as e:
        print(f"Error in detect_lip_movement: {e}")
        return False, None


def download_from_gcs(gcs_path, local_path):
    """
    Downloads a file from GCS (gcs_path) to a local path (local_path).
    """
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
    """
    Uploads a local file to GCS (gcs_path).
    """
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


def compute_crop_region(cx, cy, frame_width, frame_height, aspect_w=9, aspect_h=16):
    """
    Given a centroid (cx, cy) and the frame size (frame_width, frame_height),
    compute a 9:16 bounding box centered at (cx, cy). Clamps the box so it
    remains within the frame boundaries.
    """
    desired_aspect = aspect_w / aspect_h

    # Aim for the full frame height, with width = height * (9/16).
    new_width = int(frame_height * desired_aspect)
    new_height = frame_height

    # If new_width exceeds the frame width, adjust accordingly.
    if new_width > frame_width:
        new_width = frame_width
        new_height = int(new_width / desired_aspect)

    left = cx - new_width // 2
    right = left + new_width
    top = cy - new_height // 2
    bottom = top + new_height

    # Clamp within the frame boundaries
    if left < 0:
        left = 0
        right = new_width
    if right > frame_width:
        right = frame_width
        left = frame_width - new_width
    if top < 0:
        top = 0
        bottom = new_height
    if bottom > frame_height:
        bottom = frame_height
        top = frame_height - new_height

    return left, right, top, bottom


def process_video(filename, bucket_name="my-bucket"):
    """
    Processes the video stored in:
        gs://<bucket_name>/v-input/{filename}
    and writes the cropped (9:16) output to:
        gs://<bucket_name>/v-input/output/{filename}

    Ideal for usage in a serverless function where 'filename'
    is obtained from a POST request.
    """
    try:
        print("Starting video processing pipeline...")

        # Build GCS paths
        input_path = f"gs://{bucket_name}/v-input/{filename}"
        output_path = f"gs://{bucket_name}/v-input/output/{filename}"

        # Step 1: Download the input video from GCS
        local_input_path = os.path.join("/tmp", filename)
        download_from_gcs(input_path, local_input_path)

        # Step 2: Process the video locally
        cap = cv2.VideoCapture(local_input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {local_input_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Original video properties: FPS={fps}, W={frame_width}, H={frame_height}")

        # We initialize the VideoWriter once we know the first crop dimension
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        # Track lip centroid across frames
        prev_lip_position = None

        # Path for the final local output
        local_output_path = os.path.join("/tmp", f"output_{filename}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            # Lip detection + get centroid
            lip_moving, current_lip_position = detect_lip_movement(frame, prev_lip_position)

            if current_lip_position is not None:
                cx, cy = current_lip_position
            else:
                # If no centroid, default to frame center
                cx, cy = (frame_width // 2, frame_height // 2)

            # Compute the crop bounds for 9:16 ratio
            left, right, top, bottom = compute_crop_region(cx, cy, frame_width, frame_height, 9, 16)
            cropped_frame = frame[top:bottom, left:right]

            # Initialize the VideoWriter if needed (once we have a valid crop)
            if out is None:
                crop_h, crop_w, _ = cropped_frame.shape
                out = cv2.VideoWriter(local_output_path, fourcc, fps, (crop_w, crop_h))

            out.write(cropped_frame)

            prev_lip_position = current_lip_position

        cap.release()
        if out is not None:
            out.release()
        print("Video processing with 9:16 cropping completed successfully.")

        # Step 3: Upload the output video to GCS
        upload_to_gcs(local_output_path, output_path)

    except Exception as e:
        print(f"Error in process_video: {e}")


# Example usage (in a serverless environment, you'd call process_video from your POST handler)
#if __name__ == "__main__":
# Suppose we receive 'myvideo.mp4' from a POST request
#file_name = "myvideo.mp4"
# process_video(file_name, bucket_name="my-bucket")
