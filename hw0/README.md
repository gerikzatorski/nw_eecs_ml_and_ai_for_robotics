# Assignment 0 - Filtering Algorithms

This homework used data from the Autonomous Space Robotics Lab at the University of Toronto which can be seen here: [http://asrl.utias.utoronto.ca/datasets/mrclam/index.html](http://asrl.utias.utoronto.ca/datasets/mrclam/index.html).

## Usage

```sh
run.py [-h] [--plot] {ds0,ds1} maxsteps

positional arguments:
  {ds0,ds1}   which dataset to use
  maxsteps    maximum number of steps

optional arguments:
  -h, --help  show this help message and exit
  --plot      plot the results? (default: False)
```

## Results

### ds0 - no occlusions

![Broken Link Image](hw0/img/paths_ds0.png)
![Broken Link Image](hw0/img/errors_ds0.png)

### ds1 - occluded landmarks

![Broken Link Image](hw0/img/paths_ds1.png)
![Broken Link Image](hw0/img/errors_ds1.png)
