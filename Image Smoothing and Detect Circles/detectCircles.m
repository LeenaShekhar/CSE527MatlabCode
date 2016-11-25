clc; clear all; close all;

image = imread('im8.png'); 

%gaussian to smooth the image

gaussian = fspecial('gaussian');

image1 = conv2(gaussian, image);

 

%pre processing

image2 = (image1>100);

image3 = bwareaopen(image2, 50);

 

%finding gradient and magnitude

[magnitude,gradient] = imgradient(image3);

[rows,columns] = size(gradient);

 

%all possible values of x

x_range = (1:min(rows,columns));

 

%accumulator matrix to collect votes

accmulator = zeros(rows,columns);

 

for row =1:rows,

    for column =1:columns,

        if(gradient(row,column) ~= 0),

            slope = tand(gradient(row,column) + 90);

            intercept = column - slope*row;

            for i=1:size(x_range,2),

                y = round(slope*x_range(1,i) + intercept);

                if (1<=y && y<columns),

                    accmulator(x_range(1,i),y) = accmulator(x_range(1,i),y) + 1;

                end;

            end;

        end;

    end;

end;

 

%radius location will have the maximum votes

[maxx,maxy]=find(accmulator==max(accmulator(:)));

 

magnitude=padarray(magnitude,[128,128]);

maxx=maxx+128;

maxy=maxy+128;

[rows,columns] = size(magnitude);

sum_array = [];

 

for r = 1:min(rows,columns),

    disk_k = fspecial('disk',r);

    ring_k = imgradient(disk_k);

    ring_k = im2bw(ring_k,0);

    [kernelx, kernely] = size(disk_k);

    if(round(maxx-kernelx/2)>=1) & (round(maxy-kernely/2)>=1) & (round(maxx+kernelx/2)<rows) & (round(maxy+kernely/2)<columns),

        submatrix = magnitude(round(maxx-kernelx/2):round(maxx+kernelx/2-1),round(maxy-kernely/2):round(maxy+kernely/2-1));

        submultiplication = submatrix.*ring_k;

        current_sum = sum(submultiplication(:));

        sum_array = [sum_array; current_sum];

    end;

end;

 

%marking the centre

radii = [2];

 
flag = 0;
for i = 2:size(sum_array,1)-1,

    previous_sum = sum_array(i-1);

    next_sum = sum_array(i+1);

    if (sum_array(i) - previous_sum) > 0 & (sum_array(i) - next_sum) > 0,
        if(abs(i-flag)>5),
            flag = i;
            radii = [radii;i];
        end

    end

end

 

%marking other circles

figure();

imshow(image);

hold on

for i = 1:size(radii,1),

viscircles([maxy(1)-128,maxx(1)-128],radii(i));

end

hold off

fprintf('The center is: (%d,%d) \n', maxx(1)-128,maxy(1)-128);
fprintf('The radii are: \n')
fprintf('%d \n', radii(2:end));