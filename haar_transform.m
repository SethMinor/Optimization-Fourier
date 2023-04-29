%% Haar Transform Code
clear, clc, clf;

% Fontsize, for plotting
fs = 16;

%% Baguette example
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
