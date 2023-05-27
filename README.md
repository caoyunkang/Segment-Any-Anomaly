# Segment Any Anomaly
This repo is the official implementation of 
[Segment Any Anomaly without Training via Hybrid Prompt Regularization, SAA+](http://arxiv.org/abs/2305.10724).

SAA+ aims to segment any anomaly without any training. We meet this expectation by adapting existing foundation models, 
i.e., [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and 
[Segment Anything](https://github.com/facebookresearch/segment-anything), with hybrid prompt regularization.

## Framework
We found the simple assemble of foundation models suffer from severe language ambiguity. Hence, we introduce hybrid prompts
derived from domain expert knowledge and target image context to alleviate the language ambiguity, as follows.

![](./assets/framework.png)

## Quick Start

### Dataset Preparation

We evaluate SAA+ on four public datasets, including MVTec-AD, VisA, KSDD2, and MTD. In addition, SAA+ was a winner team
in [VAND workshop](https://sites.google.com/view/vand-cvpr23/challenge), which offers a specified dataset, i.e., VisA-Challenge. 
Please consider to prepare the datasets according to following instructions.

By default, we save the data in ``../datasets``. 

```
cd $ProjectRoot # e.g., /home/SAA
cd ..
mkdir datasets
cd datasets
```

Then following the corresponding instructions to prepare individual datasets.
- [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- [VisA-Public](https://github.com/search?q=spot+the+difference&type=repositories)
- [VisA-Challenge](https://codalab.lisn.upsaclay.fr/competitions/12499)
- [KSDD2](https://www.vicos.si/resources/kolektorsdd2/)
- [MTD](https://github.com/abin24/Magnetic-tile-defect-datasets.)


### Environment Setup
You can simply use our script one-click setup environment and download the checkpoints.
```
cd $ProjectRoot
bash install.sh
```

### Repeat the public results

**MVTec-AD**

``
python run_MVTec.py
``

**VisA-Public**

``
python run_VisA_public.py
``

**VisA-Challenge**

``
python run_VAND_workshop.py
``

The submission files can be found in ``./result_VAND_workshop/visa_challenge-k-0/0shot``.

**KSDD2**

``
python run_KSDD2.py
``

**MTD**

``
python run_MTD.py
``

## Performance
![](./assets/results.png)
![](./assets/qualitative_results.png)
## What's New


- We have updated this repo for SAA+.
- We have published [Segment Any Anomaly without Training via Hybrid Prompt Regularization, SAA+](http://arxiv.org/abs/2305.10724).


## :hammer:Todolist

We will add following features in the near future...

- [x] Update repo for SAA+
- [X] Detail the zero-shot anomaly detection framework.
- [x] Evaluate on other image anomaly detection datasets.
- [ ] Add UI for easy evaluation.
- [ ] Update Colab demo.
- [ ] HuggingFace demo.

## 💘 Acknowledgements
Our work is largely inspired by the following projects. Thanks for their admiring contribution.

- [WinClip](https://github.com/caoyunkang/WinClip)
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)


## Citation

If you find this project helpful for your research, please consider citing the following BibTeX entry.

```BibTex

@article{cao_segment_2023,
	title = {Segment Any Anomaly without Training via Hybrid Prompt Regularization},
	url = {http://arxiv.org/abs/2305.10724},
	number = {{arXiv}:2305.10724},
	publisher = {{arXiv}},
	author = {Cao, Yunkang and Xu, Xiaohao and Sun, Chen and Cheng, Yuqi and Du, Zongwei and Gao, Liang and Shen, Weiming},
	urldate = {2023-05-19},
	date = {2023-05-18},
	langid = {english},
	eprinttype = {arxiv},
	eprint = {2305.10724 [cs]},
	keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Artificial Intelligence},
}

@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@inproceedings{ShilongLiu2023GroundingDM,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Shilong Liu and Zhaoyang Zeng and Tianhe Ren and Feng Li and Hao Zhang and Jie Yang and Chunyuan Li and Jianwei Yang and Hang Su and Jun Zhu and Lei Zhang},
  year={2023}
}
```
