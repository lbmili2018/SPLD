function SPLD

    %% load data
    load MRI_AD1 fea gnd;
    cnum =5;
    num = size(fea, 1);
    gnd3d = zeros(num,3);

    %%set 
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

    Idx = kmeans(fea, cnum); % pre cluster the training data

    %% begining of training
    k = randperm(num);
    train_x = fea(k(1:500),:); % training data
    train_y = gnd3d(k(1:500),:);
    test_x = fea(k(501:end),:);
    test_y = gnd3d(k(501:end),:);
    train_Idx = Idx(k(1:500),:); % index of training data when random.

    [train_x, mu, sigma] = zscore(train_x);
    test_x = normalize(test_x, mu, sigma);
        
    %% SPLD

    opts.numepochs = 400;
    opts.update = 1.02;
    opts.update2 = 1.04;
    opts.pace2 = 0.01;
    opts.train_Idx = train_Idx;

    nn = spldtrain(nn, train_x, train_y, opts);
    [er, bad] = nntest(nn, test_x, test_y);
end