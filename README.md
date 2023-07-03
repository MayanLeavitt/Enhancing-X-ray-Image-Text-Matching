<h1 align="center"> Enhancing-X-ray-Image-Text-Matching </h1> 
<h3 align="center"> Project B 044169 – Spring 2022 </h3>
<h5 align="center"> Technion – Israel Institute of Technology </h5>

  <p align="center">
    <a href="https://github.com/MayanLeavitt"> Mayan Leavitt </a> •
    <a href="https://github.com/idankinderman"> Edan Kinderman </a> 
  </p>
  
<br />
<br />

<p align="center">
  <img src="https://user-images.githubusercontent.com/82229571/219778522-8ba040a8-011c-4158-88af-8a975237d0a8.png" />
  <img src="https://user-images.githubusercontent.com/82229571/219778775-bc5aed0d-0f45-4dee-aa12-7dae9275119f.png" />
</p>

<br />
<br />

- [Summary](#summary)
- [The SGRAF Model](#sgraf)
- [Data](#data)
- [Proposed Improvements](#proposed-improvements)
- [Comparison](#comparison)
- [Files and Usage](#files-and-usage)
- [References and credits](#references-and-credits)


<h2 id="summary"> :pencil: Summary </h2>

Our project aimed to improve the matching of two X-ray scans with their fitting radiology report, using the SGRAF image-text matching model as a baseline. To achieve this, we tested various loss functions, architectures, and training methods.

Through our experimentation, we successfully incorporated the second X-ray scan into our models and achieved significantly better results. Our research provides insights into enhancing the accuracy of image-text matching, which can have important implications for medical diagnosis and treatment.

<br />



<h2 id="sgraf"> :lungs: The SGRAF Model </h2>

After extracting image and text features, the model learns vector-based similarity representations to characterize local and global alignments. The SAF module attends on significant alignments while reducing the disturbance of less meaningful alignments. For more details see the original article [[1]](#ref1).

<p align="center">
  <img src="https://user-images.githubusercontent.com/82229571/219783657-0c6bd01b-41df-447a-a61f-542be48d6dd1.png" />
</p>

<br />


<h2 id="data"> :stethoscope: Data </h2>

We used the MIMIC-CXR dataset, which contains studies with a frontal image, a lateral image and a radiology report.
The lateral image is often not used even though it contains critical information.

<br />


<h2 id="proposed-improvements"> :thought_balloon: Proposed Improvements </h2>

1. Check different loss functions: Bi-directional ranking loss, NT-Xnet [[2]](#ref2) and their weighted sum.
2. Train two models simultaneously – one for each viewpoint, and use learned weights to average the similarity scores.
3. Concatenate the image features to obtain one input.
4. Use positional encoding to differentiate between the two viewpoints.

<br />


<h2 id="comparison"> :bar_chart: Comparison </h2>

All the matrices here are of matching image to the right text. Higher R@K means better retrival. 

Here is a comparison of the basic models, which trained only on one type of image (frontal or lateral). Those are matrices of matching inage to the right text.

| Image type        | Loss           | R@1        | R@5           | R@10        |
| ---------------- |:-----------------:| :-----------------:| :-----------------:| :-----------------:|
| Frontal | BRL | 0.5 | 4.2 | 8.5 |
| Lateral | BRL | 0.5 | 1.5 | 3.1 |
| Frontal | NT-Xent | 6.6 | 18.6 | 27.2 |
| Lateral | NT-Xent | 5.0 | 13.9 | 21.1 |
| Frontal | Sum | 3.3 | 10.4 | 15.4 |
| Lateral | Sum | 0.3 | 2 | 3.4 |

Here is a comparison of the "double" models family, which has two encoders for encoding each image type (frontal and lateral). Those models trained on both image types.

| Model type       | Learned weights  | Shared text encoder | R@1        | R@5           | R@10        |
| ---------------- |:-----------------:| :-----------------:| :-----------------:| :-----------------:| :-----------------:|
| Uniform Average | X | X | 8.1 | 21.3 | 29.3 |
| Weighted Average | X | X | 8.2 | 21.2 | 29.5 |
| Double Model | V | X | 6.7 | 21.1 | 30.4 |
| Light Double Model | V | V | 8.5 | 22.5 | 31.5 |
| Pretrained Model | V | X | 8.1 | 21 | 29.6 |

Here is a comparison of the "concatinaion" models family, which gets as input a text and a concatenation of the frontal and lateral image. Some of those models trained with positional encoding [[4]](#ref4) added to the images. 

| Model type        | Positional encoding           | R@1        | R@5           | R@10        |
| ---------------- |:-----------------:| :-----------------:| :-----------------:| :-----------------:|
| Basic Concatenation | X | 7.4 | 20.2 | 29.9 |
| Tagged Features | X | 6.6 | 20.1 | 27 |
| Constant Positional Encoding | V | 7.4 | 18.8 | 27.2 |
| Full Positional Encoding | V | 7.5 | 20.6 | 28 |

We can see that using the lateral images improves results as opposed to using frontal data alone.
In addition, training two models at once achieves the best performance, but concatenating image features is a cheaper way to combine viewpoints.

<br />



<h2 id="files-and-usage"> :man_technologist: Files and Usage</h2>

| File Name        | Description           |
| ---------------- |:-----------------:|
| average_eval.py | Evaluate 2 trained models  |
| data_xray.py | Dealing with the data loading and batching |
| evaluation_xray.py | Evaluate a trained model |
| model_xray.py | The models implementation |
| opts_xray.py | Running experiments using scripts |
| train_xray.py | For training a model |

<br />


<h2 id="references-and-credits"> :raised_hands: References and credits</h2>

* Project supervisor: Gefen Dawidowicz. Some of the algorithms were implemented based on her code.
* <a id="ref1">[[1]](https://arxiv.org/abs/2101.01368)</a> Z. M. L. Diao, "Similarity Reasoning and Filtration for Image-Text Matching", AAAI conference on artificial intelligence, 2021.
* <a id="ref2">[[2]](https://arxiv.org/abs/2002.05709)</a> K. N. H. Chen, “A Simple Framework for Contrastive Learning of Visual Representations” PMLR , pp. 1597-1607, 2020. 
* <a id="ref3">[[3]](https://pubmed.ncbi.nlm.nih.gov/35647616/)</a> S. M. S. P. G. Ji, “Improving Joint Learning of Chest X-Ray and Radiology Report by Word Region Alignment”  MLMI ,  pp. 110-119, 2021.
* <a id="ref4">[[4]](https://arxiv.org/abs/1706.03762/)</a> S. P. U. J. N. G. K. P. Vaswani, “Attention is all you need”  Advances in neural information processing systems, 2017. 
