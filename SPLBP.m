function SPLBP

load MRI_AD1 fea gnd;

num = size(fea, 1);

gnd3d = zeros(num,3);
for i = 1:1:num
    switch gnd(i,:)
        case 0
            gnd3d(i,:) = [1 0 0];
        case 1
            gnd3d(i,:) = [0 1 0];
        case 2
            gnd3d(i,:) = [0 0 1];
    end
end

inlayer = size(fea', 1);
outlayer = size(gnd3d', 1);
nn = nnsetup([inlayer 100 outlayer]);

k = randperm(num);
train_x = fea(k(1:500),:);
train_y = gnd3d(k(1:500),:);
test_x = fea(k(501:end),:);
test_y = gnd3d(k(501:end),:);


%     [train_x, ~] = mapminmax(train_x',0,1);
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% SPLBP


opts.update = 1.02;
opts.batchsize = 500;
opts.numepochs = 20;
nn = nntrain(nn, train_x, train_y, opts);
opts.numepochs = 380;
nn = spltrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
end