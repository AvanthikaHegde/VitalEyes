import cv2
from deepface import DeepFace
from collections import Counter
import json
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access variables
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(url, key)

# Start webcam capture
webcam = cv2.VideoCapture(0)
cv2.namedWindow("Emotion Detector", cv2.WINDOW_NORMAL)

emotions_log = []  # store all detected emotions
display_emotion = "Detecting..."  # persistent display emotion
frame_count = 0

stress_map = {
    "angry": "High Stress",
    "fear": "High Stress",
    "sad": "Stressed",
    "happy": "Calm",
    "neutral": "Calm",
    "surprise": "Calm",
    "disgust": "Moderate Stress"
}

while True:
    ret, frame_img = webcam.read()
    if not ret:
        print("Unable to access webcam.")
        break

    frame_count += 1
    try:
        # Detect emotion
        analysis = DeepFace.analyze(frame_img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        emotions_log.append(dominant_emotion)
        display_emotion = dominant_emotion  # update only when detected

        # Every 15 frames -> log to console + Supabase
        if frame_count % 15 == 0:
            stress_level = stress_map.get(dominant_emotion, "Unknown")

            log_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dominant_emotion": dominant_emotion,
                "stress_level": stress_level
            }

            # Print JSON to console
            print(json.dumps(log_data, indent=4))

            # Save to Supabase
            supabase.table("emotions_log").insert({
                "emotion": dominant_emotion,
                "stress_level": stress_level
            }).execute()

    except Exception as e:
        pass  # keep showing last detected emotion

    # Display persistent emotion (avoid blinking)
    cv2.putText(frame_img, f"Emotion: {display_emotion}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

    # Show live video
    cv2.imshow("Emotion Detector", frame_img)

    # Exit conditions
    pressed_key = cv2.waitKey(1) & 0xFF
    if pressed_key in [27, ord('q'), ord('Q')]:  # ESC or Q to quit
        break
    if cv2.getWindowProperty("Emotion Detector", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()

# After session ends: find most frequent emotion
if emotions_log:
    most_common_emotion = Counter(emotions_log).most_common(1)[0][0]
else:
    most_common_emotion = "undetected"

stress_level = stress_map.get(most_common_emotion, "Unknown")

final_log = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dominant_emotion": most_common_emotion,
    "stress_level": stress_level
}

# Save final dominant emotion to JSON
with open("emotion_log.json", "w") as f:
    json.dump(final_log, f, indent=4)

print("Final dominant emotion saved to emotion_log.json ✅")

# Insert final record into Supabase
supabase.table("emotions_log").insert({
    "emotion": most_common_emotion,
    "stress_level": stress_level
}).execute()

# Print last 5 entries from database
response = supabase.table("emotions_log").select("*").order("id", desc=True).limit(5).execute()
print(".....LAST 5 DATABASE ENTRIES......\n")
print(response.data)
