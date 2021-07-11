# DeepWindow
This is an offical pytorch implementation of the manuscript:

>@article{lian2020deepwindow,
  title={DeepWindow: Sliding window based on deep learning for road extraction from remote sensing images},
  author={Lian, Renbao and Huang, Liqin},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={13},
  pages={1905--1916},
  year={2020},
  publisher={IEEE}
}

In case of any questions, please do not hesitate to contact us.

### How to use:
There are two scripts with a `__main__` method:

1. `training/road_training.py`: This script will train a HourGlass model according to the training data in the `data/mass` folder. You can replace the training data to train the model for other scenes

2. `road_tracking.py`: This script will trace the road in the test image, and you can observe the tracking process if you turn on the variable of visualize_tracing.

***Important Note:** 
1. There is a weight file pretrained previously, which can be used to run the program of road tracking.

2. The are some training samples in the data directory, which demostrate the format of our training sample. You can generat your own training samples according these samples using our sampling tool `ManualPointSampling`.

3. The tracking process will be slower and slower due to the time consuming operation of plot function. You can turn off the variable of visualize_tracing to run the road tracking fast.

### Output:
1. The tracking results: When the tracking process completes, a messagebox will be poped up, and the program will save the tracked road network in the folder of `results/output` if you confirm the messagebox. There will be three files saved road mask file, overlay file and road coordinates file named as `*_ mask.png`、`*_ Over.png` and `*_ gt.npy`, respectively.

2. The trained weights: The checking point file of the model will be saved in the folder of `results/run`

### Disclaimer:
Copyright (C) 2021  Renbao Lian, Liqin Huang / Fuzhou University, Fuzhou, China


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


If our work has any inspiration for your research, please cite this our paper:

<pre>
@article{lian2020deepwindow,
  title={DeepWindow: Sliding window based on deep learning for road extraction from remote sensing images},
  author={Lian, Renbao and Huang, Liqin},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={13},
  pages={1905--1916},
  year={2020},
  publisher={IEEE}
}
</pre>
