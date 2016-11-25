
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Template Tracker %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%functon call example: trackTemplate('hw4data/CarSequence', 1, 'car_template.jpg')

function trackTemplate(path_to_car_sequence, sigma, template)
%Input: 1. path to frames diectory, 2. sigma for weight matrix 3. template
%path
%you want to track
%Output: It creates ''coordinates.txt/' file inside 'output' folder and if
%save figure function is uncommented saves images as well.
%Observation: WHen a Gaussian weight matrix with low sigma value was taken,
%its performance was similar to that of without giving more weightage to
%cenre pixels. Though, when Iincreased the sigma value too much, p_Delta
%started having very low values and became nan at some point. Also due to
%matrix multiplications so many times, runtime increased.

%all output images and files are stored inside the 'output' folder
if ~exist('output', 'dir')
  mkdir('output');
end

%collecting all car frames in frames for further processing

%imagedir = dir('/home/leena/Downloads/hw4data/hw4data/CarSequence/*.jpg');
folder_name = sprintf('%s/*.jpg',path_to_car_sequence);
imagedir = dir(folder_name);
nimages = length(imagedir);    % number of car frames in the folder
for n=1:nimages
    frame = imread(imagedir(n).name);
    images(:, :, :, n) = frame;
end

%reading the given template
template = imread(template);
template = rgb2gray(template);
[rt, ct] = size(template);

%creatin a Gaussian weight matrix to give more weightage to centre pixels.
weight_matrix = fspecial('gaussian', [rt ct], sigma);

%reading the first image and locating template using normxcorr2 for the first time in the image
image = images(:, :, :, 1);
image = rgb2gray(image);
[r, c] = size(image);
cc = normxcorr2(template, image);
[rcc, ccc] = size(cc);
[~, imax] = max(abs(cc(:)));
[ypeak, xpeak] = ind2sub(size(cc), imax(1));
br = ypeak - (rt-1);
bc = xpeak - (ct-1);

%[h, w, channels, nimages] = size(sequence);

%tracking different objects
%note: while tracking differen templates, the template is lost as the
%moving object is nomore inside the frame.
box = [bc, br, bc + ct, br + rt];
% box = [330 213 418 262]; %1. Red vintage car

%coordinates of the first image
nimage = regexp(imagedir(1).name,'\d+','match');
coord = [str2double(nimage{:}), box(1), box(2), box(1)+ct, box(2), box(1)+ct, box(2)+rt, box(1), box(2)+rt];

%to keep track of all the coodinates of the box in all the images
coordinates = zeros(nimages, 9);
coordinates(1, :) = coord;

%iterate over all the images to calculate the displacement for the box
for n = 1:nimages-1
  
    img = im2double(images(: , :, :, n));
    imshow(img);
    title('Template Tracker');
    hold on;
    
    if( ~ isnan(box(1)) &  ~isnan(box(2)))
        rectangle('Position', [box(1), box(2), ct, rt], 'LineWidth', 3, 'EdgeColor', 'm');
    end
    hold off;
    pause(.05);
    
    frame_num = str2double(nimage{:});
    nimage = regexp(imagedir(n).name,'\d+','match');

    %saving the images inside 'output' directory
    %saveImagesProduced(frame_num);
    
    %calculate the displacement vector for the box 
    It = rgb2gray(im2double(images(:, :, :, n)));
    It1 = rgb2gray(im2double(images(:, :, :, n+1)));
    
    %initializing motion vectors
    u = 0;
    v = 0;
    
    % get template image
    [Sx, Sy] = meshgrid(box(1):(box(3) - 1), box(2):(box(4) - 1));
    template = interp2(It, Sx, Sy);% 49*88

    %gaussian on template
    %template = template .* weight_matrix;
    
    % evaluate the gradient of the window in x an dy directions
    [dTx, dTy] = gradient(template);
    
    %vectorizing
    vdTx = dTx(:);
    vdTy = dTy(:);
    warpGrad = [vdTx, vdTy];
    
    sq_Tx = dTx .^ 2;
    sigma_Tx = sum(sq_Tx(:));
    
    sq_Ty = dTy .^ 2;
    sigma_Ty = sum(sq_Ty(:));
    
    product = dTx .* dTy;
    sigma_TxTy = sum(product(:));
    
    M = [sigma_Tx, sigma_TxTy; sigma_TxTy, sigma_Ty];
    
    emargin = 0.0005;
    p_delta = [1;1]';
    
    %iterate till acceptable convergence or max number of iterations
    for iterations= 1:2000
        
        [X, Y] = meshgrid((box(1)+ u) : (box(3) -1 + u), (box(2)+ v) : (box(4) + v - 1));
        
        It1_warped = interp2(It1, X, Y);% 50*89
        
        %gaussian
        [h, w] = size(It1_warped);
        gaussian = fspecial('gaussian', [h w], sigma);
        %It1_warped = It1_warped .* gaussian; 
        
        % compute error
        diff_window = It1_warped - template;
        diff_window = diff_window(:);
        
        % compute deltaP
        %disp(cond(warpGrad));
        %disp(cond(diff_window));
        p_delta = M \(warpGrad' * diff_window);
        
        % update motion vectors, u and v, in every iteration
        u = u - p_delta(1);
        v = v - p_delta(2);
        
        %check break condition
        %disp(norm(del_p));
        if (norm(p_delta) <= emargin)
            break;
        end
        
    end
    
    %locate the box in the image using displacement calculated above
    box = [box(1)+u, box(2)+v, box(3)+u, box(4)+v];
    nimage = regexp(imagedir(n+1).name,'\d+','match');
    box = round(box);
    coord = [str2double(nimage{:}), box(1), box(2), box(1)+ct, box(2), box(1)+ct, box(2)+rt, box(1), box(2)+rt];
    coordinates(n+1, :) = coord;
    
end

% box coordinates are in this sequence-> UL, UR, LR, LL
dlmwrite('output/coordinates.txt', coordinates, 'delimiter','\t','newline','pc');

%closes all the open windows, for convenience
close all;

end

function saveImagesProduced(frame_num)
    filename = sprintf('output/frame00%d.jpg', frame_num);
    set(gcf, 'name', filename);
    saveas(gcf, filename, 'jpg');
end


%written by: Leena Shekhar

