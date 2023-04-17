# UNET_Segmentation

## Abstract
CODEX, or co-detection by indexing, is an imaging approach that allows visualization of the spatial distribution of many proteins within tissue at a microscopic resolution. It uses a novel method of detecting antibody binding events using DNA barcodes and fluorescent analogs to achieve these spatial protein signals. CODEX can be used to label cells within tissue using known combinations of protein expressions specific to each cell type. This allows for deep characterization of cellular niches and dynamics important in a wide range of clinical pathology tasks, such as tumor grading. Recent studies suggest that a deep learning model, called UNET, which is frequently used to identify regions of fluid-filled lung in chest X-ray, can identify the distribution of proteins not directly detected via CODEX-generated images. Traditional cell labeling methods are time-consuming and expensive. With an innovative deep-learning algorithm, we aim to accurately estimate protein distribution within tissue samples and improve the efficiency of the cell-labeling process. We developed a UNET model to reconstruct single-channel protein signals based on the distribution of other proteins within the same microenvironment. During model testing, a total of 56 UNET models are implemented and tested, one for each protein signal. Protein signals which can be accurately reconstructed from others may be ignored in future experiments without significant loss of information. This novel model addresses the needs of researchers seeking to determine cell-level protein expression within tissue samples.

## Background
- CO-Detection by indEXing
- Highly-mutiplexed cytometric imaging 
- DNA barcodes and fluorescent analogs
-Currently used for:
    - Characterize and quantify cellular traits
    - Intercellular environment analysis
- Cell labeling:
    - Of 60+ protein signals
    - In multiple cycles
    - 1 protein → 1 channel!
 
![image](https://user-images.githubusercontent.com/91340560/232601494-6c1a4893-cb36-4d6f-b916-953a9f2f9724.png)

## Data Description
![image](https://user-images.githubusercontent.com/91340560/232601725-8ccc95d9-62fd-45e3-b0d8-7407efa554e8.png)

## U-Net Structure
![image](https://user-images.githubusercontent.com/91340560/232603237-adc0a1a3-8f2e-4747-ad78-ee6a49205c08.png)

## Method
![image](https://user-images.githubusercontent.com/91340560/232603368-b1fb3071-63d4-4fa1-a67b-300e6e7ad33b.png)

## Overview and Results
Hypothesis: We can extrapolate the distribution of non-imaged proteins using existing CODEX imaging data.
![image](https://user-images.githubusercontent.com/91340560/232603564-16b9367e-a557-4456-9275-6a69b860fc5b.png)

## Result and Impact
Impact: Researcher can save time and money by generating synthetic reconstructions of protein distributions.
![image](https://user-images.githubusercontent.com/91340560/232604390-9a03aec5-5099-4c98-8dbe-eec40b0dc7e5.png)

## Discussion
![image](https://user-images.githubusercontent.com/91340560/232604493-a343fa88-d476-4835-906f-694850c36aa5.png)






