import cv2
import mediapipe as mp
# This is vertual simmulation with nords 
def fun(data):
    print(32+95)
# Initialize mediapipe hand detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up hand detection
hands = mp_hands.Hands(
    static_image_mode=False,      # For real-time video
    max_num_hands=2,              # Detect up to 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR (OpenCV) → RGB (MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand detection
    result = hands.process(rgb)

    # If hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks and connections on the hand
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
            )

            # Get the tip of the index finger (landmark #8)
            index_tip = hand_landmarks.landmark[8]
            h, w, c = frame.shape
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # Draw a circle on the index finger tip
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            # Display coordinates
            cv2.putText(frame, f"Index: ({x},{y})", (x+20, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Show the frame
    cv2.imshow("Hand Movement Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quitting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)