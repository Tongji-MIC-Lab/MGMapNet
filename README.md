# MGMapNet: Multi-Granularity Representation Learning for End-to-End Vectorized HD Map Construction (MGMapNet-ICLR2025)

## Overview:
The construction of vectorized high-definition map typically requires capturing both category and geometry information of map elements. Current state-of-theart methods often adopt solely either point-level or instance-level representation, overlooking the strong intrinsic relationship between points and instances. In this work, we propose a simple yet efficient framework named MGMapNet (multigranularity map network) to model map elements with multi-granularity representation, integrating both coarse-grained instance-level and fine-grained pointlevel queries. Specifically, these two granularities of queries are generated from the multi-scale bird’s eye view features using a proposed multi-granularity aggregator. In this module, instance-level query aggregates features over the entire scope covered by an instance, and the point-level query aggregates features locally. Furthermore, a point-instance interaction module is designed to encourage information exchange between instance-level and point-level queries.

## Methods:
The overall architecture of MGMapNet is depicted in Fig. 1 (a). MGMapNet comprises a BEV feature encoder which is responsible to extract multi-scale BEV features from perspective view images, and a Transformer Decoder which stacks multiple layers of multi-granularity attention (MGA) to generate predictions for map elements. The prediction from each layer encapsulates both category and geometry information within the perception range. Fig. 1 (b) illustrates the l-th MGA decoder layer, which is composed of self attention, MGA, and feed-forward network. The MGA consists of two components: multi-granularity aggregator and point-instance interaction. The instance-level query is initialized using learnable parameters and updated through interaction with BEV features, while the point query is dynamically generated by aggregating BEV features. As shown in Fig. 1 (c), the point-instance interaction facilitates the mutual interaction among local geometric information, global category information, and the point queries from the (l − 1)-th layer.

<p align="center">
<image src="Figs/Fig2.png" width="650">
<br/><font>Fig. 1. Overview of the proposed MGMapNet</font>
</p>


## Result:
The proposed MGMapNet is compared with several state-of-the-art vectorized high-definition map construction methods on two public datasets. The experimental results are shown in Table 1 and Table 2. Qualitative experiments are conducted to verify the effectiveness of the proposed MGMapNet, as illustrated in Fig. 2 and Fig. 3. Across three challenging scenarios—daytime with occluded vehicles, nighttime under low-light conditions, and low-light environments with occlusion, MGMapNet demonstrates robust performance in accurately identifying key elements.

<p align="center">
<font>Table 1. Comparison with the state-of-the-art discriminative models on both VisDial v0.9 validation set and v1.0 test set.</font><br/>
<image src="Figs/table1.png" width="450">
</p>
<p align="center">
<font>Table 2. Comparison to the state-of-the-art methods on Argoverse2 val set.</font><br/>
<image src="Figs/table2.png" width="450">
</p>
<p align="center">
<image src="Figs/Fig3.png" width="450">
<br/><font>Fig. 2. Qualitative visualization on nuScenes val set.</font>
</p>
<p align="center">
<image src="Figs/Fig5.png" width="450">
<br/><font>Fig. 3. Qualitative visualization on Argoverse2 val set.</font>
</p>



## Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

### Train and Test

Train MGMapNet with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/MGA/100query_class3_110epoch.py 8
```

Eval MGMapNet with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/MGA/100query_class3_110epoch.py ./path/to/ckpts.pth 8
```




## Visualization 

we provide tools for visualization and benchmark under `path/to/MapTR/tools/maptr`

### NuScenes

Download nuScenes V1.0 full dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running


**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.


**Prepare Argoverse2 data**

*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/data_converter/av2_converter.py --data-root ./data/argoverse2/sensor/
```


## Step-by-step installation instructions
Please follow the steps below to install.
[View Installation Guide](https://github.com/hustvl/MapTR/blob/main/docs/install.md)


## Acknowledgements
We use [MapTR](https://github.com/hustvl/MapTR) as reference code.

## Citation
Please cite the following paper if you find this work useful:

Jing Yang, Minyue Jiang, Sen Yang, Xiao Tan, Yingying Li, Errui Ding, Jingdong Wang, and Hanli Wang, MGMapNet: Multi-Granularity Representation Learning for End-to-End Vectorized HD Map Construction, The Thirteenth International Conference on Learning Representations (ICLR’25), Singapore, accepted, Apr. 24 - 28, 2025.