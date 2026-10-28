from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# download the tomsmail pretrained drone detector
weights = hf_hub_download(repo_id="TomSmail/drone-yolo-v1", filename="best.pt")

model = YOLO(weights)

# run on the curated DUT sample images
results = model.predict(
    source="data/post1_sample",
    imgsz=1280,
    conf=0.25,
    device="cpu",
    save=True,
    project="runs",
    name="tomsmail_1280",
)

print("Finished!")
