%% Wavelets and image segmentation
% Fourier analysis final project
clear, clc, clf;

% Fontsize, for plotting
fs = 16;

%% Convolutional neural network
% Pooling layers --> wavelets

% Using the 'CamVid' dataset from Cambridge researchers
% This is pixel-labeled images with 32 classes (car, ped, road, ...)

% Dataset importing instructions and helper functions
% See MatLab tutorial: https://www.mathworks.com/help/vision/ug/semantic-segmentation-with-deep-learning.html

% CamVid images
imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';

% Labels
labelURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip';

% Filepaths
%outputFolder = fullfile(tempdir, 'CamVid');
outputFolder = 'C:\Users\sminor2848\Documents';
imageDir = fullfile(outputFolder,'camvid_pics\701_StillsRaw_full');
labelDir = fullfile(outputFolder,'camvid_labels');

% Download the repository
% if ~exist(outputFolder, 'dir')
%     disp('Downloading 557 MB CamVid data set...');
%     unzip(imageURL, imageDir);
%     unzip(labelURL, labelDir);
% end

% Classes and RGB encodings
classNames = [ ...
    "Animal", ...
    "Archway", ...
    "Bicyclist", ...
    "Bridge", ...
    "Building", ...
    "Car", ...
    "CartLuggagePram", ...
    "Child", ...
    "Column_Pole", ...
    "Fence", ...
    "LaneMkgsDriv", ...
    "LaneMkgsNonDriv", ...
    "Misc_Text", ...
    "MotorcycleScooter", ...
    "OtherMoving", ...
    "ParkingBlock", ...
    "Pedestrian", ...
    "Road", ...
    "RoadShoulder", ...
    "Sidewalk", ...
    "SignSymbol", ...
    "Sky", ...
    "SUVPickupTruck", ...
    "TrafficCone", ...
    "TrafficLight", ...
    "Train", ...
    "Tree", ...
    "Truck_Bus", ...
    "Tunnel", ...
    "VegetationMisc", ...
    "Wall"];

labelIDs = [ ...
    064 128 064; ... % "Animal"
    192 000 128; ... % "Archway"
    000 128 192; ... % "Bicyclist"
    000 128 064; ... % "Bridge"
    128 000 000; ... % "Building"
    064 000 128; ... % "Car"
    064 000 192; ... % "CartLuggagePram"
    192 128 064; ... % "Child"
    192 192 128; ... % "Column_Pole"
    064 064 128; ... % "Fence"
    128 000 192; ... % "LaneMkgsDriv"
    192 000 064; ... % "LaneMkgsNonDriv"
    128 128 064; ... % "Misc_Text"
    192 000 192; ... % "MotorcycleScooter"
    128 064 064; ... % "OtherMoving"
    064 192 128; ... % "ParkingBlock"
    064 064 000; ... % "Pedestrian"
    128 064 128; ... % "Road"
    128 128 192; ... % "RoadShoulder"
    000 000 192; ... % "Sidewalk"
    192 128 128; ... % "SignSymbol"
    128 128 128; ... % "Sky"
    064 128 192; ... % "SUVPickupTruck"
    000 000 064; ... % "TrafficCone"
    000 064 064; ... % "TrafficLight"
    192 064 128; ... % "Train"
    128 128 000; ... % "Tree"
    192 128 192; ... % "Truck_Bus"
    064 000 064; ... % "Tunnel"
    192 192 000; ... % "VegetationMisc"
    064 192 000];    % "Wall"

% Image sizes are 720x960x3 pixels (RGB)
camvid_datastore = imageDatastore(imageDir);
camvid_labelstore = pixelLabelDatastore(labelDir, classNames, labelIDs);

figure (1)
for image_no = 1:2
    rando = ceil(700*rand);
    subplot(2,2,image_no)
    imshow(readimage(camvid_datastore,rando))
    subplot(2,2,image_no+2)
    imshow(uint8(readimage(camvid_labelstore,rando)).^2./3)
end

% Create the training data
training_data = combine(camvid_datastore, camvid_labelstore);

% The neural network
% Pooling layers
layers = [
    % ------------------------------------------------------
    % DOWN-SAMPLING LAYERS
    % Input layer (RGB images)
    imageInputLayer([720, 960, 3])

    % First convolutional layer, convolution2dLayer(filterSize,numFilters)
    % Creates feature maps
    convolution2dLayer(3,2,'Padding','same')
    % Batch normalization
    %batchNormalizationLayer
    % Activation function
    reluLayer
    % Down-sample by a factor of 2: max pooling layer (2x2)
    maxPooling2dLayer(2,'Stride',2)
    % 360 x 480

    % Second convolutional layer
    convolution2dLayer(3,4,'Padding','same')
    %batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    % 180 x 240

    % Third convolutional layer
    convolution2dLayer(3,8,'Padding','same')
    %batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    % 90 x 120

    % Fourth convolutional layer
    convolution2dLayer(3,16,'Padding','same')
    %batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    % 45 x 60

    % ------------------------------------------------------
    % UP-SAMPLING LAYERS
    % 'Tranposed deconvolution' for up-sampling by a factor of 2
    transposedConv2dLayer(4,16,'Stride',2,'Cropping',1)
    reluLayer

    transposedConv2dLayer(4,8,'Stride',2,'Cropping',1)
    reluLayer
    
    transposedConv2dLayer(4,4,'Stride',2,'Cropping',1)
    reluLayer

    transposedConv2dLayer(4,2,'Stride',2,'Cropping',1)
    reluLayer

    % ------------------------------------------------------
    % PIXEL CLASSIFICATION LAYER
    % Classifies an image same size as the input
    % (Except that channel numbers goes from 3 --> large (num. filters)

    % Squeeze the number of feature maps to 31
    convolution2dLayer(1,31)

    % Softmax layer, turn activation values into PDF
    softmaxLayer

    % Tie each pixel to its predicted class, measure loss
    pixelClassificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs', 4, ...
    'Shuffle', 'every-epoch', ...
    'MiniBatchSize', 16, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Image segmentation model
%neural_net = trainNetwork(training_data, layers, options);

% Wavelet
wavelet_layers = [
    % ------------------------------------------------------
    % DOWN-SAMPLING LAYERS
    % Input layer (RGB images)
    imageInputLayer([720, 960, 3])

    % First convolutional layer, convolution2dLayer(filterSize,numFilters)
    % Creates feature maps
    convolution2dLayer(3,3,'Padding','same')
    % Batch normalization
    batchNormalizationLayer
    % Activation function
    reluLayer
    % Down-sample by WAVELET
    %maxPooling2dLayer(2,'Stride',2)
    functionLayer(@(X) my_haar2t(X))
    %[x_ll, x_lh, x_hl, x_hh] = haart2(G, level)
    % 360 x 480

    % Second convolutional layer
    convolution2dLayer(3,4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    %maxPooling2dLayer(2,'Stride',2)
    functionLayer(@(X) my_haar2t(X))
    % 180 x 240

    % Third convolutional layer
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    %maxPooling2dLayer(2,'Stride',2)
    functionLayer(@(X) my_haar2t(X))
    % 90 x 120

    % Fourth convolutional layer
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    %maxPooling2dLayer(2,'Stride',2)
    functionLayer(@(X) my_haar2t(X))
    % 45 x 60

    % ------------------------------------------------------
    % UP-SAMPLING LAYERS
    % 'Tranposed deconvolution' for up-sampling by a factor of 2
    transposedConv2dLayer(4,16,'Stride',2,'Cropping',1)
    reluLayer

    % Second 'Tranposed deconvolution' for up-sampling by a factor of 2
    transposedConv2dLayer(4,8,'Stride',2,'Cropping',1)
    reluLayer
    
    % Third up-sampling by a factor of 2
    transposedConv2dLayer(4,4,'Stride',2,'Cropping',1)
    reluLayer

    % Fourth up-sampling by a factor of 2
    transposedConv2dLayer(4,2,'Stride',2,'Cropping',1)
    reluLayer

    % ------------------------------------------------------
    % PIXEL CLASSIFICATION LAYER
    % Classifies an image same size as the input
    % (Except that channel numbers goes from 3 --> large (num. filters)

    % Squeeze the number of feature maps to 31
    convolution2dLayer(1,31)

    % Softmax layer, turn activation values into PDF
    softmaxLayer

    % Tie each pixel to its predicted class, measure loss
    pixelClassificationLayer];

% Image segmentation model
neural_net = trainNetwork(training_data, wavelet_layers, options);


%% Helper functions

function X_ll = my_haar2t(X)
    [X_ll, ~, ~, ~] = haart2(X, 1);
end
