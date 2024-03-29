# Generalizable COVID-19 detection network
This repository contains the code for feature extraction, model architecture and spectral-temporal saliency map described in the paper "Spectral-Temporal Saliency Masks and Modulation Tensorgrams for Generalizable COVID-19 Detection".

## Dataset
We used DiCOVA2 and ComParE 2021 COVID-19 Speech dataset. We do not re-distribute these data as both require approval from the data holders. However, they can be obtained by contacting the authors of [DiCOVA2](https://dicovachallenge.github.io/) and [ComParE](https://arxiv.org/abs/2102.13468).
Note that for ComParE, several COVID-positive recordings were with a different sampling rate, which has been shown to lead to over-optimistic results. In our experiment, we remove these recordings. If you wish to train models on the unbiased version, you may either contact us or the authors of [ComParE Challenge](https://arxiv.org/abs/2102.13468) to obtain the id of these biased files.

## Feature extraction
We use modulation tensorgram as a novel representation of speech. This representation can be extracted using a one-line code. We release the code/toolboxes for two versions of modulation tensorgrams, both were experimented in our study. Note that although the experiments in this paper showed the former was better for DiCOVA2 and ComParE, the latter together with a linear SVM achieved comparable performance to a fintuned VGGish network (which is reported in our [TASLP paper](https://ieeexplore.ieee.org/document/10097559)). Hence, we encourage interested readers to try both for your own implementation.

Code location: [SRMR toolbox](https://github.com/MuSAELab/SRMRToolbox) and [Modbank toolbox](https://github.com/MuSAELab/modulation_filterbanks)

## Model architecture
A convolutional-recurrent network is used to analyze the modulation tensorgram input. This is because modulation tensorgram is essentially a 3D representation. The 3D convolutional block is used to aggregates spatial and temporal information, while the cascaded RNN processes the temporal dynamics. The whole model is very light-weight with only 0.9 million parameters (similar to a BiLSTM if using spectrogram as input) owing to the compactness of modulation tensorgram. The model is depicted in the figure below.

Code location: [Model file](https://github.com/zhu00121/COVID-CRNN/blob/main/script/model.py) <br>
Load the trained version using [saved model parameters](https://github.com/zhu00121/COVID-CRNN/blob/main/script/model_params.pt)

## Spectral-temporal saliency map
This saliency map is designed to capture important regions in modulation tensorgram by incorporating temporal saliency. Although it is here applied to our COVID-CRNN model, the computation can be used together with any models with a CNN and RNN block.

Code location: [Saliency map](https://github.com/zhu00121/COVID-CRNN/blob/main/script/generate_saliency.py)

## Citation
Our paper is currently under review, but you may cite the [preprint version](https://www.techrxiv.org/articles/preprint/Spectral-Temporal_Saliency_Masks_and_Modulation_Tensorgrams_for_Generalizable_COVID-19_Detection/21791837)

## Contact info
If you have any questions, do not hesitate to contact me at Yi.Zhu@inrs.ca. More work on speech applications can also be found at [our lab website](https://musaelab.ca/).
