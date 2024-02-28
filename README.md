# Prototypical Bidirectional Adaptation and Learning for Cross-Domain Semantic Segmentation (PBAL)
IEEE Transactions on Multimedia 2023, [Paper](https://ieeexplore.ieee.org/document/10102322/)

Abstract
---
Cross-domain semantic segmentation, which aims to address the distribution shift while adapting from a labeled source domain to an unlabeled target domain, has achieved great progress in recent years. However, most existing work adopts a source-to-target adaptation path, which often suffers from clear class mismatching or class imbalance issues. We design PBAL, a prototypical bidirectional adaptation and learning technique that introduces bidirectional prototype learning and prototypical self-training for optimal inter-domain alignment and adaptation. We perform bidirectional alignments in a complementary and cooperative manner which balances both dominant and tail categories as well as easy and hard samples effectively. In addition, We derive prototypes efficiently from a source-trained classifier, which enables class-aware adaptation as well as synchronous prototype updating and network optimization. Further, we re-examine self-training and introduce prototypical contrast above it which greatly improves inter-domain alignment by promoting better intra-class compactness and inter-class separability in the feature space. Extensive experiments over two widely studied benchmarks show that the proposed PBAL achieves superior domain adaptation performance as compared with the state-of-the-art.

Usage
---
- Prepare Datasets: [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset as the source domain, and the [Cityscapes](https://www.cityscapes-dataset.com/) dataset as the target domain.

- Put the [warm_up model](https://drive.google.com/file/d/1xvSJnNFDCOqb73kGZbP1MB97Tvl9nUbS/view?usp=drive_link) into the 'pretrain_model' folder.

- Train stage1 ([our trained BPL model](https://drive.google.com/file/d/13pEivIotb7zHtaTZbjTbCn0niJ7tYBZu/view?usp=drive_link))
```
python train_bpl.py
```

- Train stage2 (*you should generate pseudo labels in advance using the 'stage1' trained model*, [our trained PST model](https://drive.google.com/file/d/1xbIg5JLG8iBut0NIOOR_CyBEtCUWjsue/view?usp=drive_link))
```
python train_pst.py
```

- Following the same knowledge distillation technique by [ProDA](https://github.com/microsoft/ProDA), we have the [final model](https://drive.google.com/file/d/1HtaZLhx_5WKHN9h8z7f2GnRZQEdym3hp/view?usp=drive_link).

- Inference (*you can modify the trained model in 'configs/test_from_gta_to_city.yml'*)
```
python test.py
```

The pretrained models
---
Download the pretrained [gta_to_city model](https://drive.google.com/file/d/1HtaZLhx_5WKHN9h8z7f2GnRZQEdym3hp/view?usp=drive_link).

Download the pretrained [synthia_to_city model](https://drive.google.com/file/d/1k93djzCsHn_DkeIPuKqr4RvsPyCTUaBS/view?usp=drive_link).

Citation
---
```
@article{ren2023prototypical,
  title={Prototypical Bidirectional Adaptation and Learning for Cross-Domain Semantic Segmentation},
  author={Ren, Qinghua and Mao, Qirong and Lu, Shijian},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```
Acknowledgments
---
This code is heavily borrowed from [CAG_UDA](https://github.com/RogerZhangzz/CAG_UDA) and [ProDA](https://github.com/microsoft/ProDA).
