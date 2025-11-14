# PPA - Plant Phenotyping Algorithm

PPA is a lightweight web application for plant phenotyping analysis. It leverages powerful backend algorithms and deep learning models to analyze and compute key phenotypic traits from user-uploaded plant images.

## Project Structure

```powershell
├─backend
│  ├─api
│  └─modules
│      ├─BranchAngle
│      ├─ColorScan
│      ├─GrainCount
│      ├─LeafGender
│      └─SpikeSize
└─frontend
    ├─static
    │  ├─css
    │  └─js
    └─templates
```

## Module Features

- BranchAngle: Automatically detects and calculates the angles of branches from an uploaded plant image.
- ColorScan: Analyzes a specified region of an image to extract its primary RGB color values.
- GrainCount: Identifies and counts the number of grains on a plant spike or ear from an image.
- LeafGender: Predicts the plant's gender by analyzing the morphological features of its leaves.
- SpikeSize: Measures the dimensions, such as length and width, of a plant spike or ear in an image.

## TODO

- BranchAngle: Initial algorithm draft is complete.
- LeafGender: Initial model draft is complete.
- ColorScan: Initial script draft is complete.
- GrainCount: Currently in development.
- SpikeSize: Currently in development.