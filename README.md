# 4DFlowNetv2
Refined adaptation of Super Resolution 4D Flow MRI using Residual Neural Network. This repository supplements the article https://arxiv.org/pdf/2111.11660.pdf.

This is a refinement of the [original 4DFlowNet repository](https://github.com/EdwardFerdian/4DFlowNet) using Tensorflow 2.0 with Keras. Changes include:
* Experimentation with dense and cross-partial blocks instead of residual ones
* Additional augmentation steps, such as 3D rotation with any angle and aliasing
* Grid and mask generation from CFD data 

If you are using later Tensorflow 1.x version that is not compatible with this 1.8 version, please refer to Tensorflow backwards compatibility (tf.compat module). 

## Contact Information

If you encounter any problems in using the code, please open an issue in this repository.

Author: Derek Long
