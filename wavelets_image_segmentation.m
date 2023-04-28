%% Wavelets and image segmentation
% Fourier analysis final project
clear, clc, clf;

% Fontsize, for plotting
fs = 16;

%% Haar transform
% Read in and plot test image
% (Make the row and column size even)
my_image = imread('frenchest_image.jpg');
my_image = my_image(:,1:848,:);

figure (1)
imshow(my_image)

% Plot the RGB channels
figure (2)
R = my_image(:,:,1);
G = my_image(:,:,2);
B = my_image(:,:,3);

subplot(3,1,1)
imshow(R)
title('R')

subplot(3,1,2)
imshow(G)
title('G')

subplot(3,1,3)
imshow(B)
title('B')

% Peform the Haar transform
% Use the G channel
% (Goes up to level 7 if level is not specified)
level = 1; 
[x_ll, x_lh, x_hl, x_hh] = haart2(G, level);

figure (3)
colormap gray

subplot(2,2,1)
imagesc(log(abs(x_ll).^2))
title('$\textbf{x}_{ll}$','interpreter','latex','FontSize',fs)
axis equal
set(gca,'XTick',[], 'YTick', [])
xlim([0, 424])
ylim([0, 200])

subplot(2,2,2)
imagesc(log(abs(x_hl).^2))
title('$\textbf{x}_{hl}$','interpreter','latex','FontSize',fs)
axis equal
set(gca,'XTick',[], 'YTick', [])
xlim([0, 424])
ylim([0, 200])

subplot(2,2,3)
imagesc(log(abs(x_lh).^2))
title('$\textbf{x}_{lh}$','interpreter','latex','FontSize',fs)
axis equal
set(gca,'XTick',[], 'YTick', [])
xlim([0, 424])
ylim([0, 200])

subplot(2,2,4)
imagesc(log(abs(x_hh).^2))
title('$\textbf{x}_{hh}$','interpreter','latex','FontSize',fs)
axis equal
set(gca,'XTick',[], 'YTick', [])
xlim([0, 424])
ylim([0, 200])


%% Convolution neural network example
% Digits database

% See MatLab tutorial
% "Create Simple Deep Learning Neural Network for Classification"
% https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html

% Training data location
digits_data_path = fullfile(matlabroot,'toolbox','nnet',...
    'nndemos', 'nndatasets','DigitDataset');

% Import training data
digits_data = imageDatastore(digits_data_path,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Display some images
figure (4);

rando = randperm(10000, 20);
for image_no = 1:20
    subplot(4,5,image_no);
    imshow(digits_data.Files{rando(image_no)});
end

% How many images are in each 'digit' category (0,1,..9)
%category_count = countEachLabel(training_data)

% Size of the input data (28x28x1 pixels)
%test_image = readimage(training_data,1);
%size(test_image)

% Specify how many images will be used for training
% (The rest can be used for testing the network!)
training_number = 750; % 750/1000 pics per digit category

% Randomly split into two new datastores: training and testing
[training_data, test_data] = splitEachLabel(digits_data,...
    training_number, 'randomize');

% Convolution neural network achitecture
% Define the LAYERS of the network- like a 7-layer dip, but better...
layers = [
    % Input layer
    imageInputLayer([28, 28, 1])

    % First convolutional layer, eight 3x3 filters
    % Creates eight feature maps
    % Syntax is 'convolution2dLayer(filterSize,numFilters,...)'
    convolution2dLayer(3,8,'Padding','same')

    % Max pooling layer, a stride of 2
    % Syntax is 'maxPooling2dLayer(poolSize,...)'
    maxPooling2dLayer(2,'Stride',2)

    % Second convolutional layer, 32 3x3 filters
    % Increase number of filters b/c info per unit space is increased?
    % Smaller size ALLOWS more filters at same computational cost
    convolution2dLayer(3,16,'Padding','same')

    % 'Batch-normalization', for stability and performance
    % 'Normalizes'ish the activation values, apparently this can help
    % For use between convolutional layers and activation functions
    batchNormalizationLayer

    % Activation function
    reluLayer

    % Add the standard 'fully connected layer' with output of 10 (digits)
    % Fully connected to all neurons in previous layer, (w*a + b)ish
    %    ---> Combines previous features into high-level rep.
    % Syntax is 'fullyConnectedLayer(outputSize)'
    fullyConnectedLayer(10)

    % Softmax layer, turn activation values into PDF
    % Wiki: "often used as the last activation function to normalize
    % the output of a network to a probability distribution"
    softmaxLayer

    % Assigns input to a digit (0,1,...,9)
    % Also, measures the performance (loss) of the model
    classificationLayer];

% Train the network!
% Use stochastic gradient descent with 'momentum'
% Learning rate = step-size of optimization (not ste by method?)
% Epochs = number of times run through the dataset (i.e. due to SGD batches)

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', test_data, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

neural_net = trainNetwork(training_data, layers, options);

% Final accuracy stats
% Make predictions of test data
predictions = classify(neural_net, test_data);

% What are the REAL labels of the data?
validation = test_data.Labels;

accuracy = sum(predictions == validation)/numel(validation);


%% MatLab vision database
% Pooling layers --> wavelets

% MatLab example:
%https://www.mathworks.com/help/vision/ug/getting-started-with-semantic-segmentation-using-deep-learning.html

% --- PART 1 ---
% Viewing data and pixel data

% Training data location
vision_data_path = fullfile(toolboxdir('vision'),'visiondata');

% Get images of buildings
buildings_data_path = fullfile(vision_data_path,'building');
% Pixel-labeled pictures of buildings
buildings_pix_data_path = fullfile(vision_data_path,'buildingPixelLabels');

% Import training data (needs to be 'pixel-labeled')
buildings_data = imageDatastore(buildings_data_path);

% Display some images
figure (5);

for image_no = 1:4
    subplot(2,2,image_no);
    imshow(readimage(buildings_data, image_no))
end

% Define the pixel labels
classNames = ["sky", "grass", "building", "sidewalk"];
pixelLabelID = [1 2 3 4];
pixel_labels = pixelLabelDatastore(buildings_pix_data_path,...
    classNames, pixelLabelID);

% Read in an image
Raw = readimage(buildings_data,2);
Pix = readimage(pixel_labels,2);

fig_overlay = labeloverlay(Raw, Pix);

figure (6)
imshow(fig_overlay)

% Convolution neural network
% Create the neural net

layers = [
    % ------------------------------------------------------
    % DOWN-SAMPLING LAYERS
    % Input layer (RGB images)
    imageInputLayer([32, 32, 3])

    % First convolutional layer, 32 3x3 filters
    % Creates 32 feature maps
    convolution2dLayer(3,32,'Padding',1)

    % Activation function
    reluLayer

    % Down-sample by a factor of 2: max pooling layer (2x2)
    maxPooling2dLayer(2,'Stride',2)

    % First convolutional layer, 32 3x3 filters
    % Creates 32 feature maps
    convolution2dLayer(3,32,'Padding',1)

    % Activation function
    reluLayer

    % Down-sample by a factor of 2: max pooling layer (2x2)
    maxPooling2dLayer(2,'Stride',2)

    % ------------------------------------------------------
    % UP-SAMPLING LAYERS
    % 'Tranposed deconvolution' for up-sampling by a factor of 2
    % Syntax is 'transposedConv2dLayer(filterSize,numFilters,...)'
    % 'Cropping'=1 is set so output size = 2*input size
    transposedConv2dLayer(4,32,'Stride',2,'Cropping',1)

    % Activation function
    reluLayer
    
    % Second up-sampling by a factor of 2
    transposedConv2dLayer(4,32,'Stride',2,'Cropping',1)

    % Activation function
    reluLayer

    % ------------------------------------------------------
    % PIXEL CLASSIFICATION LAYER
    % Classifies an image same size as the input
    % (Except that channel numbers goes from 3 --> large (num. filters)

    % Squeeze the number of feature maps down to 3
    convolution2dLayer(1,3)

    % Softmax layer, turn activation values into PDF
    softmaxLayer

    % Tie each pixel to its predicted class, measure loss
    pixelClassificationLayer];

% Train the neural net
% Training data location (bunch of triangles)
vision_data_path = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
triangles_data_path = fullfile(vision_data_path,'trainingImages');

% Pixel-labeled pictures ("ground truth")
triangles_label_path = fullfile(vision_data_path,'trainingLabels');

% Create an image store for triangle pics
triangles_data = imageDatastore(triangles_label_path);

% Create an image store for ground truth pixel labels
classNames = ["triangle", "background"];
labelIDS = [255, 0];

triangles_labels = pixelLabelDatastore(triangles_label_path, classNames, labelIDS);


%% Convolution neural network
% Pooling layers --> wavelets

% Using the 'CamVid' dataset from Cambridge researchers
% This is pixel-labeled images with 32 classes (car, ped, road, ...)

% Dataset importing instructions and helper functions
% See: https://www.mathworks.com/help/vision/ug/semantic-segmentation-with-deep-learning.html
























