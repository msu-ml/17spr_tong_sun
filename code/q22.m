% load USPS data, 3000 samples
load dat/USPS.mat

% visualize any image
% imshow(reshape(A(1999,:), [16, 16])')

% get principle component
pcs = pca(A);

% test a number of compression levels
npc = [10, 50, 100, 200]';
mse = zeros(size(npc, 1), 1);
img = zeros(size(npc, 1) + 1, 2, 16, 16);

for i = 1:size(npc)
    Z = A * pcs(:, 1:npc(i));       % compression
    R = Z * pcs(:, 1:npc(i))';      % reconstruction
    
    err = sum((A - R).^2, 2);       % square errors
    mse(i) = mean(err);             % mean square error
    
    % take out the first two reconstructed images
    img(i, 1, :, :) = reshape(R(1,:), [16, 16])';
    img(i, 2, :, :) = reshape(R(2,:), [16, 16])';
end
% the first two original images
img(end, 1, :, :) = reshape(A(1, :), [16, 16])';
img(end, 2, :, :) = reshape(A(2, :), [16, 16])';

% report
table(npc, mse, 'VariableNames', {'NPC', 'MSE'})

% plots
for i = 1:size(img, 1)
    for j = 1:size(img, 2)
        dat = squeeze(img(i, j, :, :));   % image data
        idx = (j-1) * size(img, 1) + i;   % image index
        subplot(size(img, 2), size(img, 1), idx)
        imshow(dat);
    end
end
