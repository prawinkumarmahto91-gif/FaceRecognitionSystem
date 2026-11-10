
import cv2

from src.recognizer import load_models, recognize_image
from src.database import load_database


def main():

    detector, recognizer = load_models()
    database = load_database()

    if not database:
        print("Register at least one person first.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started. Press Q to exit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        output, results = recognize_image(
            frame,
            database,
            detector,
            recognizer,
            threshold=0.363
        )

        cv2.imshow(
            "Face Recognition - Press Q to Exit",
            output
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
