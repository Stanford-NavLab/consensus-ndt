# Consensus NDT Odometry
### Authors
Ashwin Vivek Kanhere and Grace Xingxin Gao, Stanford University

Link to paper: [LiDAR SLAM Utilizing Normal DistributionTransform and Measurement Consensus](https://web.stanford.edu/~gracegao/publications/conference/2019//2019_ION%20GNSS_Ashwin%20Kanhere_Consenus%20NDT%20SLAM_paper.pdf)

Implementation of LiDAR odometry using the Normal Distributions Transform (NDT) and measurement consensus. Measurement consensus is used to remove potentially faulty point clouds from the odometry optimization resulting in a speed-up when compared to an equivalent naive NDT odometry optimization.

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Consensus NDT Odometry](#consensus-ndt-odometry)
    - [Authors](#authors)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)
  - [Contact](#contact)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


## Prerequisites
* Python 3.xx
* 
* [Anaconda](https://www.anaconda.com/)
* Numpy>=1.0.0, can be installed using `pip install numpy`.

<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Setup Anaconda [https://www.anaconda.com/](https://www.anaconda.com/)
2. Clone the repository
```sh
git clone https://github.com/kanhereashwin/ion-gnss-19.git
```
1. Install packages
```sh
pip install -r requirements.txt
```
4. Download dataset
```sh
bash scripts/data_downloader.sh
```

<!-- Add as many subheaders as required here -->
### Usage

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Ashwin Vivek Kanhere - akanhere {at} stanford.edu
[Navigation and Autonomous Vehicles Lab](http://web.stanford.edu/~gracegao/)
Department of Aeronautics and Astronautics
Stanford University


Project Link: [https://github.com/kanhereashwin/ion-gnss-19](https://github.com/kanhereashwin/ion-gnss-19)

## Citation
If you use this code for your research, please cite our [paper](https://web.stanford.edu/~gracegao/publications/conference/2019//2019_ION%20GNSS_Ashwin%20Kanhere_Consenus%20NDT%20SLAM_paper.pdf):

'''
@inproceedings{kanhere2019consensus,
  title={LiDAR SLAM Utilizing Normal Distribution Transform and Measurement Consensus},
  author={Kanhere, Ashwin Vivek and Gao, Grace Xingxin},
  booktitle={32nd International Technical Meeting of the Satellite Division of the Institute of Navigation, ION GNSS},
  pages={2228 - 2240},
  year={2019}
}
'''


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
Code borrows the ICP implementation from [https://github.com/ClayFlannigan/icp/blob/master/icp.py](https://github.com/ClayFlannigan/icp/blob/master/icp.py) 
We also acknowledge Siddarth Tanwar and Shubh Gupta for their comments and reviews.
