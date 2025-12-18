% Quaternion-based Multi-branch Attention Network for HyperspectralImage Classification

%    This demo shows the QMAN model for hyperspectral image classification.
%
%    main.py ....... A main script executing experiments upon IP, PU, BT, and SV data sets. 
%    data.py ....... A script implementing various data manipulation functions.
%    util.py ....... A script implementing the sample selection function and etc.
%    QMAN.py ....... A script implementing the QMAN models.                        
%    train_test.py ....... A script implementing the training function and test function etc. 
%	 generatedq.py ....... A script for generating quaternions            
%    quaternion_layers.py ....... A script implementing the quaternion layers. 
%    quaternion_ops.py ....... A script containing various quaternion operations

%   --------------------------------------
%   Note: Required core python libraries
%   --------------------------------------
%   1. python 3.7
%   2. torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 
%   3. gdal
%   4. einops
%   5. numpy==1.21.6
%   6. matplotlib==3.3.0
%   7. scikit-learn==1.0.2
%   8. scipy==1.7.3
%   9. spectral==0.23.1
%   10. opencv-python==4.4.0.46
%   11. tifffile


