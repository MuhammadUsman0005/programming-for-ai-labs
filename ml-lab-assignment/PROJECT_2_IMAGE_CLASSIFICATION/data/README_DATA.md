
# Project 2 dataset setup

The professor's three-page assignment brief names the required classes but does not provide a dataset URL or attach the images. Ask the professor/TA for the official image dataset location if you have not already received it. Do not substitute unrelated images and do not invent class labels.

Put image files into exactly these folders (subfolders are also searched):

```text
data/images/
  Cars/
    car_001.jpg
    car_002.jpg
  Cricket Balls/
    ball_001.jpg
    ball_002.jpg
  Ice-Cream Cones/
    cone_001.jpg
    cone_002.jpg
```

Accepted extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. The script converts every image to grayscale and resizes it to 64 × 64 pixels. It then creates a stratified train/test split. Each class needs at least two readable images; more diverse real images will make the evaluation more meaningful.
