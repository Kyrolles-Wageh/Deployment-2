from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data=r"C:\Users\kc17062026\Desktop\Play with git\Hands on the Wheel.v4i.yolo26\data.yaml",
        epochs=40,
        imgsz=512
    )

if __name__ == "__main__":
    main()