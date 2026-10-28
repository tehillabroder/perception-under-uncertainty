from pathlib import Path
from time import perf_counter

from huggingface_hub import hf_hub_download
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# use the same weights as used for tomsmail
weights = hf_hub_download(repo_id="TomSmail/drone-yolo-v1", filename="best.pt")

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=weights,
    confidence_threshold=0.25,
    image_size=640,
    device="cpu",
)

images = sorted(Path("data/post1_sample").glob("*.jpg"))

for image in images:
    start = perf_counter()

    """
    SAHI supports two approaches:
    - sliced-only inference, which uses predictions from patches
    - sliced plus whole-image inference, which combines predictions from patches and 
        the original whole image
    The default is True, where SAHI performs whole-image prediction
    For now I chose False, because I initially want to isolate the effect of slicing
    """
    result = get_sliced_prediction(
        str(image),
        model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        perform_standard_pred=False,
        verbose=0,
    )

    elapsed = perf_counter() - start

    result.export_visuals(export_dir="runs/sahi_640", file_name=image.stem)

    print(f"{image.name}: {len(result.object_prediction_list)} predictions, {elapsed:.2f} seconds")
