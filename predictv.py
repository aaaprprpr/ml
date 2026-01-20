from ultralytics import YOLO

def run_video():
    model=YOLO("./runs/pose/train24/weights/best.pt")
    results=model.predict(
        source="./real44.mp4",
        save=True,
        show=True,
        conf=0.4,
    )

if __name__=="__main__":
    run_video()
