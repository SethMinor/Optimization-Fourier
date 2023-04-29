%% Wavelets and image segmentation
% Fourier analysis final project
clear, clc, clf;

% Fontsize, for plotting
fs = 16;

%% Convolution neural network
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

% Image sizes are 960 x 720 pixels
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
trainingData = combine(camvid_datastore, camvid_labelstore);

% The neural network
layers = [
    % ------------------------------------------------------
    % DOWN-SAMPLING LAYERS
    % Input layer (RGB images)
    imageInputLayer([960, 720, 3])

    % First convolutional layer, 2 3x3 filters
    % Creates 32 feature maps
    convolution2dLayer(3,2,'Padding',1)
    % Activation function
    reluLayer
    % Down-sample by a factor of 2: max pooling layer (2x2)
    maxPooling2dLayer(2,'Stride',2)

    % Second convolutional layer, 4 3x3 filters
    % Creates 32 feature maps
    convolution2dLayer(3,4,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    % Third convolutional layer, 8 3x3 filters
    % Creates 32 feature maps
    convolution2dLayer(3,8,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    % ------------------------------------------------------
    % UP-SAMPLING LAYERS
    % 'Tranposed deconvolution' for up-sampling by a factor of 2
    transposedConv2dLayer(3,8,'Stride',2,'Cropping',1)

    % Activation function
    reluLayer
    
    % Second up-sampling by a factor of 2
    transposedConv2dLayer(3,4,'Stride',2,'Cropping',1)

    % Activation function
    reluLayer

    % Third up-sampling by a factor of 2
    transposedConv2dLayer(3,2,'Stride',2,'Cropping',1)

    % Activation function
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

opts = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle', 'every-epoch', ...
    'MiniBatchSize',64, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Image segmentation model
%neural_net = trainNetwork(training_data, layers, options);

% Final accuracy stats
% Make predictions of test data
%predictions = classify(neural_net, test_data);

% What are the REAL labels of the data?
%validation = test_data.Labels;

%accuracy = sum(predictions == validation)/numel(validation);
