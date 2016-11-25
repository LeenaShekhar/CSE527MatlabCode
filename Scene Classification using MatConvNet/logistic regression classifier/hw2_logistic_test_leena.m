%load test features
load('feat_test.mat'); %cnn feat (.90938)
%load('spatialTstD'); % %spatial feat (.67813)
%load('tstD'); %bow feat (.59250)

test_num_img = 1600;

%reshaping the cnn feat
test_num_img = size(feat_test(2:end,:),1); %drop first row of zeros
feat_size = size(feat_test(2:end,:),2);
feat_test = reshape(feat_test(2:end,:), [1,1,cnn_feat_size,test_num_img]);

%reshaping the spatial feat
%feat_test = 5000*spatialTstD;
%feat_size = 21000;
%feat_test = single(reshape(feat_test(1:end,:), [1,1,21000,1600]));

%reshaping the bow feat
%feat_test = single(5000*tstD);
%feat_size = 1000;
%feat_test = single(reshape(feat_test(1:end,:), [1,1,bow_feat_size,test_num_img]));

%network details
net = load('data/leena/bow/exp3/trained_network.mat') ;
net.layers{3} = struct('type', 'softmax') ;

%creating predictions
prediction = single.empty(0);
for i=1:test_num_img
    disp(i);
    im = feat_test(1,1,:,i); 
    %im = (im) - net.imageMean);
    res = vl_simplenn(net, im) ;
    scores = squeeze(gather(res(end).x)) ;
    [maxScore, predlabel] = max(scores) ;
    prediction = [prediction; predlabel];

%     img_name = sprintf('%06d',tstIds(i));
%     current_filename = strcat('CV-HW2/hw2data/bigbangtheory/', img_name,'.jpg');
%     current_image = imread(current_filename);
%     figure(i) ; clf ; imagesc(current_image) ; axis equal off ;
%     title(sprintf('%s (%d), score %.3f',...
%               'test', predlabel, maxScore), ...
%       'Interpreter', 'none');

end

%generating csv file 
headers = {'ImgId', 'Prediction'};
fid = fopen('data/predictions/bow/predictions_leena_bow_1.csv', 'w') ;
fprintf(fid, '%s,', headers{1});
fprintf(fid, '%s', headers{2});
for i=1:size(tstIds,1)
    fprintf(fid, '\n%d,%d', tstIds(i,1), prediction(i,1));
end
fclose(fid);





