# Genetic Algorithm (MNIST + KDTree Fitness)

This repository contains a small project that:
1) downloads and saves a subset of MNIST digit images locally,  
2) performs basic image analysis / preprocessing,  
3) computes a KDTree-based “similarity” score for images, and  
4) runs a Genetic Algorithm (GA) that tries to evolve a sequence of 10 digit-images whose labels match the first 10 digits of π: **[1, 4, 1, 5, 9, 2, 6, 5, 3, 5]**.

## Repository structure

- `genetic_algorithm.py`  
  Main script (contains all functions + `main()`).
- `mnist_images/`  
  Output folder created by the script (stores the first 500 MNIST images as PNGs).
- `report.pdf`  
  Project report / write-up.

## What the code does

### Session 1 (Data + inspection)
- **Download MNIST** using `sklearn.datasets.fetch_openml`
- Save the **first 500** images as PNG files into `mnist_images/`
- Analyze a few images (shape, pixel count, intensity range)
- Crop images by removing empty (all-zero) rows/columns
- Plot **log-scaled histograms** for first occurrences of each digit label

### Session 2 (KDTree scoring + Genetic Algorithm)
- Compute a **KDTree-based score** for each image:
  - `classify_images_with_kdtree`: binary score based on nearest-neighbor distance thresholding
  - `classify_images_with_kdtree_improved`: continuous similarity score, computed *within the same class* only
- Run a **Genetic Algorithm**:
  - An individual = 10 selected images (plus their labels + KDTree scores)
  - Fitness = matches to π digits (**pi_quality**) + KDTree quality contribution
  - Also includes an “improved” fitness variant that weights KDTree quality

Outputs include:
- Fitness evolution plots (best/avg π match and best/avg KDTree score)
- A plot of the best individual’s 10 images in a grid with predicted label vs target π digit

## Requirements

Python 3.9+ recommended.

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn pillow scipy loguru
```

Notes:
- The first MNIST download can take a bit (it uses OpenML caching via scikit-learn).
- Matplotlib windows will open during execution.

## How to run

From the repository root:

```bash
python genetic_algorithm.py
```

This will:
1) download MNIST and create `mnist_images/` if it doesn’t exist,  
2) generate plots for analysis/histograms,  
3) compute KDTree vectors (basic + improved),  
4) run the GA twice (baseline + improved), producing evolution plots and best-individual visualizations.

## Key parameters (in `genetic_algorithm()`)

You can tweak:
- `num_generations = 100`
- `population_size = 50`
- `individual_length = 10`
- `mutation_rate = 0.2`
- `pi_digits = [1,4,1,5,9,2,6,5,3,5]`

## Notes / known behaviors

- `mnist.target` labels come as strings from OpenML; the code often casts to `str(...)` for comparison consistency.
- The GA selection strategy is elitist: it keeps the top 2 and repeatedly breeds from them.
- Cropping is implemented but (currently) histogram/GA runs on the original `images` array, not the cropped set.

## Report

See `report.pdf` for methodology, experiments, and results.
