% load USPS data, 3000 samples
load dat/USPS.mat

% visualize any image
% imshow(reshape(A(1999,:), [16, 16])')

% get principle component
pcs = pca(A);

% test a number of compression levels
npc = [5, 10, 20, 50, 100, 200]';
mse = zeros(size(npc));
for i = 1:size(npc)
    Z = A * pcs(:, 1:npc(i));       % compression
    R = Z * pcs(:, 1:npc(i))';      % reconstruction
    
    err = sum((A - R).^2, 2);       % square errors
    mse(i) = mean(err);             % mean square error
end

% report
table(npc, mse, 'VariableNames', {'NPC', 'MSE'})
