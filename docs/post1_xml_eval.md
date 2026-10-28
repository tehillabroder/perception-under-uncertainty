
00015.jpg: [{'label': 'UAV', 'box': [537.0, 289.0, 594.0, 325.0]}]
00018.jpg: [{'label': 'UAV', 'box': [467.0, 233.0, 521.0, 267.0]}]
00028.jpg: [{'label': 'UAV', 'box': [716.0, 277.0, 748.0, 308.0]}]
00039.jpg: [{'label': 'UAV', 'box': [614.0, 325.0, 638.0, 337.0]}]
00043.jpg: [{'label': 'UAV', 'box': [753.0, 295.0, 818.0, 329.0]}]
00059.jpg: [{'label': 'UAV', 'box': [531.0, 427.0, 554.0, 439.0]}]
00110.jpg: [{'label': 'UAV', 'box': [762.0, 265.0, 838.0, 295.0]}]
00152.jpg: [{'label': 'UAV', 'box': [192.0, 352.0, 240.0, 377.0]}]
00176.jpg: [{'label': 'UAV', 'box': [718.0, 433.0, 801.0, 476.0]}]
00301.jpg: [{'label': 'UAV', 'box': [320.0, 277.0, 968.0, 565.0]}]
00347.jpg: [{'label': 'UAV', 'box': [528.0, 253.0, 663.0, 390.0]}]
00374.jpg: [{'label': 'UAV', 'box': [379.0, 105.0, 632.0, 286.0]}]
00605.jpg: [{'label': 'UAV', 'box': [380.0, 388.0, 453.0, 454.0]}, {'label': 'UAV', 'box': [369.0, 228.0, 478.0, 314.0]}, {'label': 'UAV', 'box': [244.0, 41.0, 403.0, 191.0]}]
00667.jpg: [{'label': 'UAV', 'box': [878.0, 368.0, 1038.0, 457.0]}]
00967.jpg: [{'label': 'UAV', 'box': [244.0, 139.0, 860.0, 367.0]}]
00975.jpg: [{'label': 'UAV', 'box': [394.0, 28.0, 765.0, 291.0]}]
01020.jpg: [{'label': 'UAV', 'box': [280.0, 338.0, 558.0, 449.0]}]
01023.jpg: [{'label': 'UAV', 'box': [452.0, 179.0, 628.0, 334.0]}]
01039.jpg: [{'label': 'UAV', 'box': [567.0, 176.0, 789.0, 258.0]}]
01315.jpg: [{'label': 'UAV', 'box': [569.0, 253.0, 692.0, 294.0]}]
01411.jpg: [{'label': 'UAV', 'box': [348.0, 108.0, 560.0, 282.0]}]
01424.jpg: [{'label': 'UAV', 'box': [453.0, 315.0, 622.0, 474.0]}]
01428.jpg: [{'label': 'UAV', 'box': [181.0, 294.0, 549.0, 461.0]}]
01444.jpg: [{'label': 'UAV', 'box': [65.0, 161.0, 149.0, 199.0]}]
01451.jpg: [{'label': 'UAV', 'box': [1591.0, 353.0, 1773.0, 437.0]}]
01475.jpg: [{'label': 'UAV', 'box': [1428.0, 560.0, 1523.0, 599.0]}]
01499.jpg: [{'label': 'UAV', 'box': [544.0, 835.0, 625.0, 873.0]}]
01523.jpg: [{'label': 'UAV', 'box': [1290.0, 409.0, 1347.0, 439.0]}]
01535.jpg: [{'label': 'UAV', 'box': [732.0, 457.0, 769.0, 473.0]}]
01543.jpg: [{'label': 'UAV', 'box': [178.0, 554.0, 230.0, 575.0]}]

30 images, 32 annotated objects

I've first got the XML parser working:
- 30 images, 32 annotated UAVs.
- 00605 contains three drones. The other images contain one each.
- All annotations use the label UAV.
- Prev manual observations of 27 correct and 5 missed also added up to 32.
- Object sizes vary considerably. 00967 has a 616×228 pixel box, 00039 has a 24×12 pixel box. Must check how much these shrink during preprocessing later.


Now I need to measure whether a predicted bounding box actually matches the annotated drone.

IoU (Intersection over Union) measures the overlap between two boxes.
- IoU = 1 means identical boxes.
- IoU = 0 means no overlap.
- IoU more than or equal to 0.5 will be our initial critera for a correct detection.

Before counting true positives and false positives, we need confidence that our overlap calculation is correct.

I added a function for box_iou(a, b), and tested it with test_post1_iou.py. I will now take the actual YOLO predictions and match them against these XML boxes, makign sure each annotated drone can only be matched once.