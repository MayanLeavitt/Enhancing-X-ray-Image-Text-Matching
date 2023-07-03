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
- [The Best Model](#best)
- [Final Results](#final-results)
- [Files and Usage](#files-and-usage)
- [References and credits](#references-and-credits)


<h2 id="summary"> :pencil: Summary </h2>

Our project aimed to improve the matching of two X-ray scans with their fitting radiology report, using the SGRAF image-text matching model as a baseline. To achieve this, we tested various loss functions, architectures, and training methods.

Through our experimentation, we successfully incorporated the second X-ray scan into our models and achieved significantly better results. Our research provides insights into enhancing the accuracy of image-text matching, which can have important implications for medical diagnosis and treatment.

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)


<h2 id="sgraf"> :lungs: The SGRAF Model </h2>

After extracting image and text features, the model learns vector-based similarity representations to characterize local and global alignments. The SAF module attends on significant alignments while reducing the disturbance of less meaningful alignments.

<p align="center">
  <img src="https://user-images.githubusercontent.com/82229571/219783657-0c6bd01b-41df-447a-a61f-542be48d6dd1.png" />
</p>

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="data"> :stethoscope: Data </h2>

Studies contain a frontal image, a lateral image and a radiology report.
The lateral image is often not used even though it contains critical information.

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="proposed-improvements"> :thought_balloon: Proposed Improvements </h2>

1. Check different loss functions: Bi-directional ranking loss, NT-Xnet and their weighted sum.
2. Train two models simultaneously – one for each viewpoint, and use learned weights to average the similarity scores.
3. Concatenate the image features to obtain one input.
4. Use positional encoding to differentiate between the two viewpoints.

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="comparison"> :bar_chart: Comparison </h2>


<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="best"> :100: The Best Model </h2>

Two networks that train concurrently using the same text encoder. The outputs are averaged with learned weights.

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="final-results"> :stethoscope: Final Results </h2>

Using the lateral images improves results as opposed to using frontal data alone.
In addition, training two models at once achieves the best performance, but concatenating image features is a cheaper way to combine viewpoints.

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="files-and-usage"> :man_technologist: Files and Usage</h2>

| File Name        | Description           |
| ---------------- |:-----------------:|
|  |  |

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="references-and-credits"> :raised_hands: References and credits</h2>

* Project supervisor: Gefen Dawidowicz. Some of the algorithms were implemented based on her code.
