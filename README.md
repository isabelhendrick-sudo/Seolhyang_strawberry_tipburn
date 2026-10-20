# Tipburn in Seolhyang Strawberries: Treatment Comparison and Leaf Image Classifier

Data: AI Hub facility-crop dataset (Seolhyang strawberry, normal and tipburn groups), farm AI22_003, 1 November – 7 December 2022.

**Research question:** Do Seolhyang strawberry plants grown under different nutrient treatments show different amounts of tipburn, and can a simple model detect tipburn from leaf photos?

## Setup

```
pip install -r requirements.txt
```

```
tipburn_project/
├── analyse_tipburn.py
├── train_classifier.py
├── requirements.txt
└── data/
    ├── normal_labels/     TL_ and VL_ zips for 01.정상
    ├── tipburn_labels/    TL_ and VL_ zips for 04.잎끝마름
    ├── normal_images/     VS_ zip for 01.정상
    └── tipburn_images/    VS_ zip for 04.잎끝마름
```

## Run

```
python analyse_tipburn.py --labels data/normal_labels data/tipburn_labels
python train_classifier.py --images data/normal_images data/tipburn_images
```

## Outputs (`outputs/`)

| File | Content |
|---|---|
| fig01_dataset_overview.png | Plants, images and leaf annotations per group |
| fig02_symptoms_by_group.png | Symptomatic leaves per plant by group, with 95% CIs |
| fig03_symptoms_over_time.png | Weekly symptom trend per group |
| fig04_plant_heatmap.png | Weekly symptom proportion for every plant |
| fig05_growth_vs_symptoms.png | Growth measurements vs symptoms |
| fig06_sensor_check.png | Sensor fields stored in the labels |
| fig07_example_leaves.png | Example symptomatic and healthy leaves |
| fig08_colour_features.png | Colour features of symptomatic vs healthy leaves |
| fig09_classifier_performance.png | Confusion matrix and precision–recall curves |
| fig10_performance_by_group.png | Classifier scores within each group |
| fig11_error_examples.png | Example classification errors |
| stats.txt | Group statistics (Kruskal–Wallis, Mann–Whitney vs Normal) |
| classifier_results.txt | Classifier metrics |
