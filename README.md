# Spr22_CS256_GroupA_Motion-Style-Transfer

## Group A
****
## Summary
The ever increasing number of multimedia applications requires in-depth research in the field of  Image Generation, Domain Adaptation and Style Transfer. Users' reliance on video conferencing applications because of the latest remote working model has given rise to new opportunities to be innovative in these fields. The progress that has been made so far using the state of the art algorithms has been immense as compared to the classical image processing approaches. We have implemented and adapted a neural network model to synthesize a talking head upon receiving a stream of input frames based on the solution proposed in [2]. The proposed model works efficiently on the live stream thus enabling its applications in video conferencing. The proposed solution takes in a source image as an input followed by a live stream of frames to drive the motion and provides us with the desired output. The trained model was optimized using static quantization to make it 1.5 times faster than before.
****
## Instructions
1. If you want to run for live stream go to /Code/Implementation and follow the instructions there.
2. For training and demo go to /Code/Implementation/One-Shot and follow the instructions there.

****
## References

1. Joon Son Chung, Arsha Nagrani, and Andrew Zisserman. “VoxCeleb2: deep speaker recognition”. In INTERSPEECH, 2018.
2. Wang, T.-C., Mallya, A., & Liu, M.-Y. (2021). “One-Shot Free-View neural talking-head synthesis for video conferencing”. CVPR.
3. Martin Heusel, et al, “GANs trained by a two time-scale update rule converge to a local nash equilibrium”. In NeurIPS, 2017 References
4. Davis E. King. “Dlib-ml: A machine learning toolkit”. JMLR, 2009
5. P. Burt and E. Adelson, "A multiresolution spline with application to image mosaics," ACM Transactions on Graphics, vol. 2, (4), pp. 217-236, 1983. . DOI: 10.1145/245.247.
6. L. A. Gatys, A. S. Ecker and M. Bethge, "Image style transfer using convolutional neural networks," in 2016. doi: 10.1109/CVPR.2016.265.
7. M. Mirza and S. Osindero, “Conditional generative adversarial nets,” arXiv preprint arXiv:1411.1784, 2014.
8. P. Isola, et al, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1125–1134.
9. Langr, J., & Bok, V. (2019). “GANs in Action: deep learning with generative adversarial networks”. Manning. https://books.google.com/books?id=HojvugEACAAJ
10. Zhu, J.-Y., et al, “Unpaired image-to-image translation using cycle-consistent adversarial networks”. Computer Vision (ICCV), 2017 IEEE International Conference On.
11. R. Xu et al, "face transfer with generative adversarial network," 2017.
12. M. Liu, T. Breuel and J. Kautz, "Unsupervised image-to-image translation networks," 2017.
13. T. Karras, S. Laine, and T. Aila, “A Style-Based generator architecture for generative adversarial networks,” presented at the - 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 4396–4405, doi: 10.1109/CVPR.2019.00453.
14. Chong. Min, and Forsyth. D.A, “JoJoGAN: one shot face stylization”, 2021  doi: arXiv:2112.11641
15. Pinkney, J.N., Adler, D.: “Resolution dependent gan interpolation for controllable image synthesis between domains”. arXiv preprint arXiv:2010.05334 (2020) 
16. Siarohin, Aliaksandr, et al. "First order motion model for image animation." Advances in Neural Information Processing Systems 32, 2019
17. T.-C. Wang, A. Mallya, and M.-Y. Liu, “One-Shot Free-View neural talking-head synthesis for video conferencing,” presented at the - 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 10034–10044, doi: 10.1109/CVPR46437.2021.00991.
****

## About

The folder structure for the submission is - 

1. Reports - Project report for milestone 1 and the final report is uploaded here.
2. Presentation Slides - The slides created for all three milestone are uploaded in this folder. 
3. Diagrams - This folder includes all the architecture and implementation diagrams.
4. Basics - This folder contains the code written to understand the basics of neural style transfer and DCGAN. 
5. Code - All the folders, files and data for implementation of this project is uploaded in this folder.  
6. Results - This folder includes all the videos and images generated post code execution.

This repository is the Group A submission for the final project of the course CS256: Topics in Artificial Intelligence, at San Jose State University. 

The course is led by - Professor Mashhour Solh.

The group members are: Ajinkya Rajguru, Branden Lopez, Indranil Patil, Rushikesh Padia, Warada Kulkarni
