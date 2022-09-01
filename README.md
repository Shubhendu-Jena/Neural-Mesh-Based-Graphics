# Neural Mesh-Based Graphics
Shubhendu Jena, Franck Multon, Adnane Boukhayma.<br/> 
[ECCV Workshop 2022 CV4Metaverse](https://arxiv.org/abs/2208.05785)<br/> 
![render](https://user-images.githubusercontent.com/12934176/186115183-14c9dcc6-92f7-456a-9835-fac225fd78eb.png)

# Install
Run this command to install python environment using conda:
```
source scripts/install_deps.sh
```
Please modify the `cudatoolkit` version in the bash script above according to your system compatibilities.
# Data Preparation
Download the [Tanks and Temples](http://vladlen.info/papers/tanks-and-temples.pdf) data as in [FreeViewSynthesis.](https://github.com/isl-org/FreeViewSynthesis)<br/>
The data directory structure should follow the below hierarchy.
```
Neural-Mesh-Based-Graphics  
|-- data  
|   |-- ibr3d_tat
|   |-- split_meshes
|   |-- Checkpoints_Truck
|   |-- Checkpoints_Train
|   |-- Checkpoints_Playground
|   |-- Checkpoints_M60
```
To download split meshes for all scenes, see
[Split Meshes.](https://drive.google.com/file/d/1-_GUrVzhiznX39jIEInZsuCVywUFkJKL/view?usp=sharing)<br/>
To copy the split and processed meshes, run 
```
python scripts/copy_t_and_t.py
```
# Pretrained Models/Checkpoints
Pretrained models/Checkpoints can be found [here.](https://drive.google.com/drive/folders/1CYmV9Opm_ZqycXm1YQymU2JMFSEPFNbT?usp=sharing)<br/>
# Evaluation
To start the evaluation please run the following command:
```
python train.py --config configs/test_example.yaml --pipeline nmbg.pipelines.p3d.TexturePipeline --eval
```
# Training
To start a single scene or full training please run the following command:
```
python train.py --config configs/train_example.yaml --pipeline nmbg.pipelines.p3d.TexturePipeline --train
```
with the appropriate `paths_file` (`configs/paths_example_small_train.yaml` for single scene and `configs/paths_example_full_train.yaml` for full training) in `train_example.yaml`.
# Acknowledgements
Parts of the code were based on the original [Neural Point-Based Graphics](https://arxiv.org/abs/1906.08240) [implementation](https://github.com/alievk/npbg), [Stable View Synthesis](https://arxiv.org/abs/2011.07233) [implementation](https://github.com/isl-org/StableViewSynthesis) and [NeRF-SH](https://arxiv.org/abs/2103.14024) [implementation.](https://github.com/sxyu/plenoctree)

