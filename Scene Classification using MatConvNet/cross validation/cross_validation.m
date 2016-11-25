
function avg_accuracy = cross_validation(varargin)

%load the respective training data
load('feat_leena.mat');
%load('trD.mat');
%load('trD_spatial.mat');
%load training labels
load('trIds.mat');
load('trLbs.mat');

trD = feat(2:end, :)'; %neglecting the zeros for CNN
%feat = trD; %Bow
%feat = trD; %3-level Spatial 
rng('default');
rng(0) ;

imdb = {};
imdb.meta.classes = '12345678';
imdb.meta.sets = {'train'};

trainOpts.batchSize = 50 ;
trainOpts.numEpochs = 100 ;
trainOpts.continue = true ;
trainOpts.gpus = [] ;
trainOpts.learningRate = 0.001 ;
trainOpts.weightDecay = 0.0005 ;
trainOpts.momentum = 0.9 ;

accuracy1 = get_accuracy('data/submission/cross_val1', 1, 2, 3, 4, 5, trD, trLbs);
accuracy2 = get_accuracy('data/submission/cross_val2', 2, 3, 4, 5, 1, trD, trLbs);
accuracy3 = get_accuracy('data/submission/cross_val3', 3, 4, 5, 1, 2, trD, trLbs);
accuracy4 = get_accuracy('data/submission/cross_val4', 4, 5, 1, 2, 3, trD, trLbs);
accuracy5 = get_accuracy('data/submission/cross_val5', 5, 1, 2, 3, 4, trD, trLbs);

avg_accuracy = (accuracy1 + accuracy2 + accuracy3 + accuracy4 + accuracy5)/5;

%disp (avg_accuracy); //.3128 CNN, .3132 BoW, .2004 Spatial Pyramid

%********get accuracy for each fold*************
function accuracy = get_accuracy(export_dir, trId1, trId2, trId3, trId4, tstId, trD, trLbs, varargin)

ids = ml_kFoldCV_Idxs(size(trD,2), 5); %using the given function to generate different parts

train_ids = cat(2, ids{trId1},ids{trId2},ids{trId3},ids{trId4});   
train_data = single.empty(0);
train_label = single.empty(0);

test_ids = ids{tstId};
test_data = single.empty(0);
test_label = single.empty(0);

%********train data**********
for i = 1:size(train_ids,2)
    train_data(:,i) = trD(:,train_ids(i));
    train_label(1,i) = trLbs(i,1);
end

%*******test data***********
for i = 1:size(test_ids,2)
    test_data(:,i) = trD(:,train_ids(i));
    test_label(1,i) = trLbs(i,1);
end

classes = [1;2;3;4;5;6;7;8];
classes = reshape(classes, [1,1,8]);

trainOpts.expDir = export_dir;
trainOpts.numEpochs = 30 ;
trainOpts = vl_argparse(trainOpts, varargin);
imdb_train = prepare_data(train_ids', train_data, train_label');
imdb_test = prepare_data(test_ids', test_data, test_label');
feat_size = size(trD,1);
net = define_net(feat_size, classes);
net = train(net, imdb_train, trainOpts);
prediction = test(imdb_test, net);
accuracy = sum(test_label == prediction) / numel(test_label);

%********get data in batches for 5-fold validation*********
function [im, labels] = getBatch(imdb, batch)
channels = size(imdb.images.data,3);
im = imdb.images.data(1,1,:,batch) ;
im = reshape(im, 1, 1, channels, []) ;
labels = imdb.images.label(1,batch) ;

%********prepare imdb for the classifier***************
function imdb = prepare_data(train_ids, train_data, train_labels)

channels = size(train_data,1);
n = size(train_data,2);
imdb.images.id = single(train_ids'); 
imdb.images.data = single(reshape(train_data, [1,1,channels,n]));
imdb.images.label = single(train_labels'); 
imdb.images.set = single(ones(1,n));
imageMean = median(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data %- imageMean ;
save('imdb.mat', 'imdb');

%************do prediction***********
function prediction = test(imdb, net)

test_data = imdb.images.data;
net.layers{2} = struct('type', 'softmax') ;
channels = size(test_data,1);
n_files =  size(test_data,2); %size(test_feat,1);
%tstD = single(reshape(tstD, [1,1,channels,n_files]));

prediction = single.empty(0);
for i=1:n_files

    im = test_data(1,1,:,i); 
    %im = 256*(im - net.imageMean) ;
   
   % Apply the CNN to the larger image
    res = vl_simplenn(net, im) ;
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    prediction = [prediction; best];
    %disp(best);
end

%*********train the model**********
function net = train(net, imdb, trainOpts)

[net,~] = cnn_train(net, imdb, @getBatch, trainOpts) ;
net.layers(end) = [] ;
%net.imageMean = imageMean ;

function net = define_net(channels, classes)

f=1/100 ; %to reduce value of randomly initialized weights
rho = 5 ;
kappa = 0 ;
alpha = 1 ;
beta = 0.5 ;
normalization_params = [rho kappa alpha beta];
net = {};
net.layers{1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,channels,8, 'single'), zeros(1,8,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{2} = struct('type', 'lrn', ...
                        'param', normalization_params);

net.layers{3} = struct('type', 'softmaxloss', ...
        'class', classes);
    


