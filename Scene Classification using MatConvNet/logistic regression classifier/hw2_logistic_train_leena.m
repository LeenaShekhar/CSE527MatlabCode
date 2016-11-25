function hw2_train_leena(varargin)

%load train features
%load('feat_leena.mat'); %cnn feat
%load('spatialTrD.mat'); %spatial feat
load('trD'); %bow feat
load('trIds.mat');
load('trLbs.mat');

%cnn_feat = feat(2:end, :)';
%spatial_feat = 5000*spatialTrD;
%spatial_feat = spatialTrD;
cnn_feat = 5000*trD;

cnn_feat_size = size(cnn_feat, 1);
num_train_img = size(cnn_feat, 2);
num_classes = 8;

rng('default');
rng(0) ;
classes = [1;2;3;4;5;6;7;8];
classes = reshape(classes, [1,1,length(classes)]);

%creating imdb objects for further training
imdb = {};
imdb.images.id = single(trIds');
imdb.images.data = single(reshape(cnn_feat, [1,1,cnn_feat_size,num_train_img]));
imdb.images.label = single(trLbs');
imdb.images.set = single(ones(1,num_train_img));
imdb.meta.classes = '12345678';
imdb.meta.sets = {'train'};
%imageMedian = median(imdb.images.data(:));
imdb.images.data = imdb.images.data; % - imageMedian ;
save('imdb.mat', 'imdb');

%network and other parameters
f=1/100 ;
rho = 5 ;
kappa = 0 ;
alpha = 1 ;
beta = 0.5 ;
normalization_params = [rho kappa alpha beta];

net = {};

net.layers{1} = struct('type', 'conv', ...
    'weights', {{f*randn(1,1,1000,length(classes), 'single'), zeros(1,length(classes),'single')}}, ...
    'stride', 1, ...
    'pad', 0) ;

net.layers{2} = struct('type', 'lrn', ...
                        'param', normalization_params);
              
net.layers{3} = struct('type', 'softmaxloss', ...
    'class', classes);

trainOpts.batchSize = 50 ;
trainOpts.numEpochs = 30 ;
%trainOpts.continue = true ;
trainOpts.gpus = [] ;
trainOpts.learningRate = 0.001 ;
trainOpts.momentum = 0.9 ;
trainOpts.weightDecay = 0.0005 ;
trainOpts.expDir = 'data/leena/bow/exp3' ; 
trainOpts = vl_argparse(trainOpts, varargin);

[net,~] = cnn_train(net, imdb, @getBatch, trainOpts);
net.layers(end) = [] ;
%net.imageMean = imageMedian ;
save('data/leena/bow/exp3/trained_network.mat', '-struct', 'net') ;

function [im, labels] = getBatch(imdb, batch)
im = imdb.images.data(1,1,:,batch);
%im = reshape(im, 1, 1, 4096, []); %cnn feat  
%im = reshape(im, 1, 1, 21000, []); %spatial feat  
im = reshape(im, 1, 1, 1000, []); %bow feat 

labels = imdb.images.label(1,batch);