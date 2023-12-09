
# Genetic Algorithm for Image Reconstruction

## Overview
This project implements a genetic algorithm for image reconstruction. It consists of multiple Python modules, each contributing to the evolution of a population of images towards a target image using natural selection principles.

## Modules
- `crossover.py`: Manages the crossover operation between pairs of images.
- `evaluationfunctions.py`: Provides functions for evaluating the fitness of each image.
- `genetic_algorithm_for_image_reconstruction.py`: Orchestrates the genetic algorithm process.
- `methods.py`: Includes various auxiliary methods such as initialization and selection strategies.
- `mutations.py`: Defines mutation operations for image modification.
- `population.py`: Handles the population management of images.
- `selector.py`: Implements the selection logic for choosing images for crossover and mutation.
- `main.py`: The entry point of the application.

## Installation
To run this project, ensure you have Python 3.x installed. Clone the repository and install the required packages using:
```
pip install -r requirements.txt
```

## Usage
To start the image reconstruction process, run the following command:
```
python main.py
```

## Configuration
Modify the parameters of the genetic algorithm (like population size, mutation rate, etc.) in `main.py`.

