# Post 1: Initial Detection Baseline

Model: TomSmail/drone-yolo-v1 (pretrained YOLOv8n)  
Dataset: 30 selected images from DUT Anti-UAV test set  
Hardware: Intel Mac, CPU inference  
Confidence threshold: 0.25  
Input sizes: 640 and 1280

## Results

| Metric | 640 | 1280 |
|---|---|---|
| Missed drones | 6 | 4 |
| Incorrect/extra predictions | 5 | 12 |
| Extra predictions per image | 0.17 | 0.40 |

Manual observations, not yet validated against ground-truth annotations. The 1280 count includes one duplicate detection.

## 640 observations

Missed drones:
- `00043`: two incorrect quadcopter predictions, actual drone missed
- `00152`: difficult to spot
- `11076`: blurry
- `00967`: large drone missed
- `01499`: very difficult to spot
- `01523`: very difficult to spot, leaf incorrectly detected

Other false positives:
- `01543`: streetlight and unrelated structure

## 1280 observations

Missed drones:
- `00043`: difficult target
- `00152`: moderately difficult target
- `01475`: moderately difficult target
- `01523`: missed despite higher resolution

Incorrect/extra predictions:
- `00015`: small blob in sky
- `00110`: duplicate detection of same drone
- `00176`: two small blobs, drone correctly detected
- `00347`: drone-shaped logo
- `00605`: bird, alongside three correctly detected drones
- `01411`: pylon, drone correctly detected
- `01428`: tractor, drone correctly detected
- `01499`: leaves, drone correctly detected
- `01523`: two leaf detections
- `01543`: streetlamp, drone correctly detected

## Initial observations

Increasing input size from 640 to 1280 reduced missed detections from 6 to 4, but increased incorrect/extra predictions from 5 to 12.

Some missed drones were large or difficult to distinguish from their background, suggesting that downsampling alone may not explain detection failures.

Higher resolution appears to expose additional background features that the model mistakes for drones.

These findings are preliminary. We still need to inspect ground-truth bounding-box dimensions, verify prediction matching and measure inference latency.

## Next

1. Verify counts against XML annotations and confirm image ID `11076`.
2. Record target sizes and confidence scores.
3. Run SAHI with the same model and images.
4. Compare recall by object size, false positives and CPU inference latency.