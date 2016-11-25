%load the trained network
net = load('imagenet-vgg-verydeep-16.mat') ;
net = vl_simplenn_tidy(net) ;

%directory path to load images
image_dir = 'hw2data/bigbangtheory/';

%***********extract train features**************%
train = load('train.mat');
train_id = train.imIds;
train_label = train.lbs;
nfiles = length(train_id);

for ii=1:nfiles
   img_name = sprintf('%06d',train_id(ii));
   current_filename = strcat(image_dir, img_name,'.jpg'); 
   current_image = imread(current_filename);
   images{ii} = current_image;
end

ph = 224;

feat = zeros(1, net.layers{36}.size(3));
for ii=1:nfiles
    disp(ii)
    image_resized = imresize(images{ii},[256 454]);
    patches = get_patches_from_image(image_resized, ph);
    ntimes = size(patches,4);
    res_sum = single(zeros(1,net.layers{36}.size(3)));
    for jj=1:ntimes
        res = vl_simplenn(net, patches(:,:,:,jj));
        repr = res(36).x;
        repr = reshape(repr, [1,net.layers{36}.size(3)]);
        res_sum = res_sum + repr;
    end
    res_sum = res_sum./ntimes;
    feat = [feat; res_sum];
end
save('cnn_feat_train.mat','cnn_feat_train')


%***********extract test features**************%
test = load('test.mat');
test_id = test_id.imIds;
nTestFiles = length(test_id);

for ii=1:nTestFiles
   img_name = sprintf('%06d',test_id(ii));
   current_filename_test = strcat(image_dir, img_name,'.jpg'); 
   current_image_test = imread(current_filename_test);
   images_test{ii} = current_image_test;
end

ph = 224;

feat_test = zeros(1, 4096); %net.layers{36}.size(3)
for ii=1:nTestFiles
    disp(ii)
    image_resized = imresize(images_test{ii},[256 454]);
    patches_test = get_patches_from_image(image_resized, ph); %function to extract patches
    ntimes = size(patches_test,4);
    res_sum = single(zeros(1,4096));
    for jj=1:ntimes
        res = vl_simplenn(net, patches_test(:,:,:,jj));
        repr = res(36).x;
        repr = reshape(repr, [1,4096]);
        res_sum = res_sum + repr;
    end
    res_sum = res_sum./ntimes;
    feat_test = [feat_test; res_sum];
end
save('cnn_feat_test.mat','cnn_feat_test')
