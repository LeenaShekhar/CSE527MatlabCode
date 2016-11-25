i = imread('cat_small.jpg');
i = im2single(i);

%visualize the inpur image
figure(100) ; clf ; imagesc(i) ; colormap gray ;

%sobel operator h and v
sx = fspecial('sobel');
sy = sx';

wsx = single(repmat(sx, [1,1,1]));
wsy = single(repmat(sy, [1,1,1]));
w = cat(4, wsx, wsy);

%pad 0, default
y_sob = vl_nnconv(i,w,[]);
figure(1) ; clf ; vl_imarraysc(y_sob) ; colormap gray ;

%stride 1, 2, and 3
y_ds1 = vl_nnconv(i, w, [], 'stride', 1);
figure(2) ; clf ; vl_imarraysc(y_ds1) ; colormap gray ;

y_ds2 = vl_nnconv(i, w, [], 'stride', 2) ;
figure(3) ; clf ; vl_imarraysc(y_ds2) ; colormap gray ;

y_ds3 = vl_nnconv(i, w, [], 'stride', 3) ;
figure(4) ; clf ; vl_imarraysc(y_ds3) ; colormap gray ;

%pad 5, and 10
y_pad5 = vl_nnconv(i, w, [], 'pad', 5);
figure(5) ; clf ; vl_imarraysc(y_pad5) ; colormap gray ;

y_pad10 = vl_nnconv(i, w, [], 'pad', 10);
figure(6) ; clf ; vl_imarraysc(y_pad10) ; colormap gray ;


%Non linear

y_relu = vl_nnrelu(y_sob) ;
figure(7) ; clf ; vl_imarraysc(y_relu) ; colormap gray ;

%Pooling

%Maxpooling 2, 5 and 15

y_pool2 = vl_nnpool(y_relu, [2,2]) ;
figure(8) ; clf ; vl_imarraysc(y_pool2); colormap gray ;

y_pool5 = vl_nnpool(y_relu, [5,5]) ;
figure(9) ; clf ; vl_imarraysc(y_pool5) ; colormap gray ;

y_pool13 = vl_nnpool(y_relu, [13,13]) ;
figure(10) ; clf ; vl_imarraysc(y_pool13) ; colormap gray ;

%Average Maxpooling 2, 5 and 13

y_av_pool2 = vl_nnpool(y_relu, [2,2], 'method', 'avg') ;
figure(11) ; clf ; vl_imarraysc(y_av_pool2); colormap gray ;

y_av_pool5 = vl_nnpool(y_relu, [5,5], 'method', 'avg') ;
figure(12) ; clf ; vl_imarraysc(y_av_pool5) ; colormap gray ;

y_av_pool13 = vl_nnpool(y_relu, [13,13], 'method', 'avg') ;
figure(13) ; clf ; vl_imarraysc(y_av_pool13) ; colormap gray ;


