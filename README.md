# Generative_Vision
Contains a project done on Generative Vision for CS 614 at Drexel University

Title: Augmenting Satellite Imagery for Airplane Detection Using Stable Diffusion Inpainting

Created By: Shawn Oyer on 7/26/2024

Abstract:  The increasing volume and complexity of satellite imagery data necessitate advanced techniques for automated object detection and segmentation. This  project explores the application of Stable Diffusion,  specifically Inpainting models for augmenting satellite imagery data, particularly focusing on enhancing object detection of airplanes. Traditional data aug mentation methods often fall short in scenarios in volving specific object localization and diverse back grounds. Leveraging Stable Diffusion’s Inpainting capabilities, this project aims to generate synthetic images by retaining the position of objects-of-interest while varying backgrounds or objects within predefined masks. The methodology includes using pretrained models, generating synthetic images through various prompts, and fine-tuning with Textual Inversion. Frechet Inception Distance (FID) and Structural Similarity Index (SSIM), were employed as evaluation metrics to assess the quality of the generated images. Results indicate that while fine-tuning improved background variation, the quality of the generated airplanes could still be enhanced. The study demonstrates that Stable Diffusion Inpainting can significantly contribute to augmenting satellite imagery for object detection tasks, although further refinement in creating more realistic objects is required.

Data: The data can be downloaded here:
https://drive.google.com/file/d/10oakKU-DFtDWGPDiTs2x7IqZ82FfwauN/view 

The data consists of the following:
Real Training Images - 5825 images in .PNG format
Real Test Images - 2710 images in .PNG format
Synthetic Training Images - 2000 images in .PNG format
Real and Synthetic Masks in JSON format

Model: The model can be downloaded here: https://github.com/shawnoyer/Generative_Vision 

The model consists of the VAE, Text Encoder, Tokenizer, Unet, Scheduler, Safety Checker, Feature Extractor, Learned Embeds, and Model Index

Contents: The contents of the .ipynb file are separated into the following sections: 

  Utilities
  Testing Examples
  Data Setup
  Stable Diffusion Inpainting Pipeline
  Stable Diffusion Fine-Tuning (Textual Inversion) Pipeline
  Results
 
Stakeholders: Used by imagery analysts, earth scientists, data scientists, professionals within Defense, Environmental, Transportation, and Agricultural Sectors

Uses and Intensions: Specifically, the project aims at two kinds of augmentations to be able to generate thousands of new synthetic images using Stable Diffusion Inpainting with satellite imagery:

  1. Retain an object and generate different backgrounds.
  2. Retain the background and generate different objects inside the mask.

Using the Script: The extension of the script is .ipynb so it can be accessed and run within jupyter notebook and exported as a .py to export into any Python IDE

Contributors and Contact List: Shawn Oyer - Drexel University Gradate Student, sbo33@drexel.edu

License Information: No License

Sources:

Aayushibansal, M., et al. “A Systematic Review on Data Scarcity Problem in Deep Learning: Solution and Applications.” ACM, ACM, 2020, https://d1wqtxts1xzle7.cloudfront.net/88792779/3502287-libre.pdf?1658317987=&response-content-disposition=inline%3B+filename%3DA_Systematic_Review_on_Data_Scarcity_Pro.pdf&Expires=1721643624&Signature=Sc0fetxjIDD8thH8Mpxg4H6IEMPsHHfRU~3SaydhsEmuPoQEJTfJldB. Accessed 22 July 2024.
Arsalan, Tahir, et al. “Automatic Target Detection from Satellite Imagery Using Machine Learning.” NCBI, PubMed Central, 2 February 2022, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8839603/. Accessed 22 July 2024.
Brownlee, Jason. “How to Implement the Frechet Inception Distance (FID) for Evaluating GANs - MachineLearningMastery.com.” Machine Learning Mastery, 11 October 2019, https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/. Accessed 23 July 2024.
Cheng, Wei Loon. “How to Build Your Own AI-Generated Image with ControlNet and Stable Diffusion.” Datature, 22 May 2023, https://www.datature.io/blog/how-to-build-your-own-ai-generated-image-with-controlnet-and-stable-diffusion. Accessed 22 July 2024.
Cole, Rob M. “satellite-image-deep-learning/techniques: Techniques for deep learning with satellite & aerial imagery.” GitHub, 8 July 2024, https://github.com/satellite-image-deep-learning/techniques. Accessed 22 July 2024.
Groener, Austen, et al. “A Comparison of Deep Learning Object Detection Models for Satellite Imagery.” Arxiv, Arxiv, 10 September 2020, https://arxiv.org/pdf/2009.04857. Accessed 22 July 2024.
Hugging Face. “Inpainting.” Inpainting, Hugging Face, 2024, https://huggingface.co/docs/diffusers/v0.29.2/en/using-diffusers/inpaint#inpainting. Accessed 22 July 2024.
Hugging Face. “Textual Inversion.” Hugging Face, 2024, https://huggingface.co/docs/diffusers/training/text_inversion. Accessed 23 July 2024.
İlaslan, Düzgün. “Stable Diffusion-Inpainting Using Hugging Face Diffusers with Serving Gradio.” Medium, 18 June 2023, https://medium.com/@ilaslanduzgun/stable-diffusion-inpainting-using-hugging-face-diffusers-with-serving-gradio-b7b4939a4888. Accessed 23 July 2024.
Imatest. “SSIM: Structural Similarity Index.” Imatest, 2024, https://www.imatest.com/docs/ssim/. Accessed 23 July 2024.
Koneripalli, Kaushik. “Satellite Image Data Augmentation using Stable Diffusion for Object detection & segmentation.” Medium, 2 September 2023, https://medium.com/@kaushik.koneripalli/satellite-image-data-augmentation-using-stable-diffusion-for-object-detection-segmentation-8b1fe87b969. Accessed 22 July 2024.
Kumari, Priyanka. “Inpainting with Stable Diffusion: Step-by-Step Guide.” Lancer Ninja, 10 October 2023, https://lancerninja.com/inpainting-stable-diffusion/. Accessed 22 July 2024.
Stable Diffusion Art. “How does Stable Diffusion work?” Stable Diffusion Art, 9 June 2024, https://stable-diffusion-art.com/how-stable-diffusion-work/. Accessed 22 July 2024.
