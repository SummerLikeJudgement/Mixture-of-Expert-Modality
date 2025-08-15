# EMOE: Modality-Specific Enhanced Dynamic Emotion Experts
> EMOE: Modality-Specific Enhanced Dynamic Emotion Experts,            
> Yiyang Fang, Wenke Huang, Guancheng Wan, Kehua Su, Mang Ye
> *CVPR, 2025*, [Link](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html)

<div align="center">
<img alt="method" src="image/EMOE.png">
</div>

## News
* [2025-06-02] Code has been released.
* [2025-05-02] Repo created. Code will be released soon.

## Abstract
<div align="justify">
Multimodal Emotion Recognition (MER) aims to predict human emotions by leveraging multiple modalities, such as vision, acoustics, and language. However, due to the heterogeneity of these modalities, MER faces two key challenges: modality balance dilemma and modality specialization disappearance. Existing methods often overlook the varying importance of modalities across samples in tackling the modality balance dilemma. Moreover, mainstream decoupling methods, while preserving modality-specific information, often neglect the predictive capability of unimodal data. To address these, we propose a novel model, Modality-Specific Enhanced Dynamic Emotion Experts (EMOE), consisting of: (1) Mixture of Modality Experts for dynamically adjusting modality importance based on sample features, and (2) Unimodal Distillation to retain single-modality predictive ability within fused features. EMOE enables adaptive fusion by learning a unique modality weight distribution for each sample, enhancing multimodal predictions with single-modality predictions to balance invariant and specific features in emotion recognition. Experimental results on benchmark datasets show that EMOE achieves superior or comparable performance to state-of-the-art methods. Additionally, we extend EMOE to Multimodal Intent Recognition (MIR), further demonstrating its effectiveness and versatility.
</div>

## Citation
```bibtex
@inproceedings{fang2025emoe,
    title     = {EMOE: Modality-Specific Enhanced Dynamic Emotion Experts},
    author    = {Fang, Yiyang and Huang, Wenke and Wan, Guancheng and Su, Kehua and Ye, Mang},
    booktitle = {CVPR},
    year      = {2025}
}
```
