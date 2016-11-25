function [moving_image] = SubtractDominantMotion(It, It1)

%Input: This function takes 2 images- image at time t and t+1
%Output: Moving image given these two images as input

%dimension of the image for further reference
[height,width] = size(It1);

M = affineMotion(It, It1);
%disp(M);

tform = affine2d(transpose(M));
It_warped = imwarp(It, tform, 'OutputView', imref2d(size(It1)), 'FillValues', 0);

dIt = abs(It1 - It_warped)/255;
filter = getFilterForCommonRegion(height, width, It, M);
dIt = dIt .* filter;

moving_image = hysthresh(dIt, .5, .3);
moving_image = medfilt2(moving_image);

moving_image = doOptimizations(moving_image) ;
moving_image = double(moving_image);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%% Helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function M = affineMotion(It, It1)
    
It = mat2gray(It);
It1= mat2gray(It1);

%initializing few parameters
emargin = .006;
p = zeros(6,1);

[height,width] = size(It1);

%iterations after which convergence
for iterations=1:2000
    
    %calclate M matrix
    affine_params = p';
    M = [affine_params(1)+1, affine_params(2), affine_params(3); affine_params(4), affine_params(5)+1, affine_params(6); 0, 0, 1];
    %M_inv = inv(M);
    %disp (M)
    
    %warping image It1 by taking It as reference
    It1_warped = getWarpedImage(height, width, It1, It, M);
        
    %matrix of temportal derivates only for common regions between It and
    %warped It1
    filter = getFilterForCommonRegion(height, width, It1_warped, M);
    dIt = It - It1_warped;
    dIt = dIt .* filter;
    
    %matrix of image derivaives only for common regions between It and warped It1
    [dIx, dIy] = getGradientForImage(height, width, It, M);
    
    %image derivative matrix
    A = calculateA(height, width, dIx, dIy);
    dIt = dIt(:);
    del_p = -(A' * A) \ (A' * dIt);
    p = p + del_p;
    
    %disp(norm(del_p));
    if (norm(del_p) < emargin)
        break;
    end    
end

end


%to compute pixels lying in the region common to im1 and warped im2
function filter = getFilterForCommonRegion(height, width, image, M)
    
%     %creating custom mask by iterating over the warped image- idea 1
%     filter = zeros(height, width);
%     for idx = 1:numel(image)
%         if (image(idx) ~= 0) 
%             fiter(idx) = 1;
%         end
%     end
%     filter = logical(mask);

    %idea is to create a filter with all zeros then 'and'ing it and 'or'ing
    %it- idea 2
    %optimizing idea 1 as this was slow
    filter = zeros(height, width);
    [wX, wY] = findWarpedPoints(height, width, M);
    filter = filter | (wY >=1 & wY <= height);
    filter = filter & (wX >=1 & wX <= width);
    
end

function [wX, wY] = findWarpedPoints(height, width, M)
    
      %idea 1 
%     U = zeros(height, width);
%     V = zeros(height, width);
%     for i = 1:height
%         for j = 1:width
%             warped = M * [j; i; 1];
%             U(i, j) = warped(1);
%             V(i, j) = warped(2);
%         end
%     end
%     wX = U;
%     wY = V;

     % we know that in affine motion u = ax + by + c and v = dx + ey + f
     %idea 2- optmizing idea 1 as this was slow
    [X, Y] = meshgrid(1:width, 1:height);
    wX = M(1,1) * X;
    wX = wX + Y * M(1,2);
    wX = wX + M(1,3);
    
    wY = X * M(2,1);
    wY = wY + Y * M(2,2);
    wY = wY + M(2,3);
    
end

function [dIx, dIy] = getGradientForImage(height, width, image, M)
    
    [dIx, dIy] = gradient(image);

    filter = getFilterForCommonRegion(height, width, image, M);
    dIx = dIx .* filter;
    dIx = dIx(:);
    %dIx = dIx/255;
    dIy = dIy .* filter; 
    %dIy = dIy/255;
    dIy = dIy(:);

end

function warped_image = getWarpedImage(height, width, Iwarp, Iref, M)
    
    tform = affine2d(transpose(M));
    %disp(tform.T);
    warped_image = medfilt2(imwarp(Iwarp, tform, 'OutputView', imref2d(size(Iref)), 'FillValues', 0));
    
    %code for custom warping but slower
%     warped_image = zeros(size(image));
%     
%     for i=1:height
%         for j=1:width
%             
%             dimension = M \ [j, i, 1]';
%             x = floor(dimension(2,1) / dimension(3,1));
%             y = floor(dimension(1,1) / dimension(3,1));
%             if(x >= 1 && y >= 1 && x <= height && y <= width)
%                 warped_image(i, j) = image(x, y); 
%             end
%         end
%     end
    
    %figure, imshow(double(warped_image));
 
end

function A = calculateA(height, width, dIx, dIy)

    [X, Y] = meshgrid(1:width, 1:height);
    X = X(:);
    Y = Y(:);
    
    a = X.*dIx;
    b = Y.*dIx;
    c = dIx;
    d = X.*dIy;
    e = Y.*dIy;
    f = dIy;
    A = [a, b, c, d, e, f];

end

function moving_image = doOptimizations(moving_image)

    se = strel('disk',10);
    moving_image = imdilate(moving_image, se);
    moving_image = imerode(moving_image, se);
    
end


%written by: Leena Shekhar- discussed with Pratik Vaishnavi and Gaurav
%Mishra

%References:
%1. Lecture slides
%2. http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf
%3. http://16720.courses.cs.cmu.edu/lec/alignment.pdf
