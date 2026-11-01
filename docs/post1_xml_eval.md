# Post 1: Evaluating YOLO and SAHI Against Ground Truth

## What I've done

I started by manually reviewing 30 deliberately selected images from the DUT Anti-UAV test set. I noticed that increasing resolution and using SAHI could recover some drones, but also introduced false positives.

To move beyond visual impressions and measure the results, I parsed the dataset's XML annotations, which confirmed that my 30 images contain 32 annotated UAVs. Image `00605` contains three drones, while the others contain one each.

I then ran the same pretrained TomSmail YOLO detector using three approaches:

- YOLO at 640
- YOLO at 1280
- SAHI with 640 × 640 slices and 20% overlap, without full-image inference

I kept the confidence threshold at 0.25 and used IoU ≥ 0.5 to decide whether a predicted bounding box matched an annotated drone. Each drone could only be matched once.

## Initial results

| Metric | YOLO 640 | YOLO 1280 | SAHI 640 |
|---|---:|---:|---:|
| Correct matches | 22 | 24 | 19 |
| False positives | 9 | 20 | 39 |
| False negatives | 10 | 8 | 13 |
| Recall | 68.8% | 75.0% | 59.4% |
| Precision | 71.0% | 54.5% | 32.8% |

Increasing YOLO's resolution from 640 to 1280 recovered more drones, but also more than doubled the false positives.

SAHI performed worse at the chosen IoU threshold, which was surprising given that I had visually seen it detect many of these drones.

## Something didn't add up

When I inspected `00605`, I could clearly see SAHI had detected all three drones and a bird.

But the evaluation counted only two correct detections.

The third drone had an IoU of 0.486, just below my 0.5 threshold. The model had recognised the drone, but its bounding box was too large.

I checked other images and found similar problems.

In `00967`, SAHI produced two overlapping boxes covering different parts of one large drone. In `00110`, `01039` and `01424`, it found the drones but produced inaccurate boxes.

Meanwhile, `01499` showed a different problem: one correctly detected drone alongside seven background false positives.

This made me realise that I needed to distinguish between finding a drone and accurately locating it.

## Checking the IoU threshold

I repeated the evaluation at three IoU thresholds, using the same predictions.

| IoU | YOLO 640 recall | YOLO 1280 recall | SAHI recall |
|---|---:|---:|---:|
| 0.30 | 75.0% | 78.1% | 75.0% |
| 0.50 | 68.8% | 75.0% | 59.4% |
| 0.75 | 28.1% | 25.0% | 6.2% |

SAHI's recall improved considerably when I relaxed the matching threshold, but it dropped sharply when I required more precise bounding boxes.

Even at IoU 0.3, SAHI still produced 34 false positives, compared with 7 for YOLO 640.

So inaccurate localisation seems to be part of the problem, and background false alarms remain a separate issue.

## Does object size explain it?

I also grouped the drones by their approximate size after whole-image resizing to 640.

| Object size | YOLO 640 | YOLO 1280 | SAHI 640 |
|---|---:|---:|---:|
| Under 16 px | 4/6 | 3/6 | 3/6 |
| 16–31 px | 4/9 | 6/9 | 5/9 |
| 32–63 px | 3/4 | 3/4 | 3/4 |
| 64+ px | 11/13 | 12/13 | 8/13 |

SAHI didn't improve recall for the smallest objects in this sample. It also struggled to produce sufficiently accurate boxes for some much larger drones.

Size clearly matters, but it doesn't seem to explain everything.

## What I'm taking away

My original question was whether preserving more pixels makes small objects easier to detect.

So far, I've seen that finding an object and accurately localising it are different problems.

Higher resolution can improve recall while creating more false alarms. Slicing can help identify objects, but it can also produce fragmented or oversized boxes.

## What I want to investigate next

I want to separate real background false positives from duplicate boxes and localisation failures.

I'll also investigate how changing the confidence threshold affects recall and false alarms, and whether combining SAHI with full-image inference helps recover accurate boxes.

Finally, I want to measure inference time consistently and repeat the evaluation on a larger, less selectively chosen set of images.

The broader question I'm now interested in is not just whether a detector sees a drone, but whether its output is accurate and reliable enough to use.