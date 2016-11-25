classdef HW2_BoW    
% Practical for Visual Bag-of-Words representation    
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 18-Dec-2015
% Last modified: 27-Sep-2016   
    
    methods (Static)
        function main()
            scales = [8, 16, 32, 64];
            normH = 16;
            normW = 16;
            bowCs = HW2_BoW.learnDictionary(scales, normH, normW);
            %save('bowCs.mat','bowCs');
            
            [trIds, trLbs] = ml_load('../bigbangtheory/train.mat',  'imIds', 'lbs');             
            tstIds = ml_load('../bigbangtheory/test.mat', 'imIds'); 
            
                        
            %[spatialTrD, hist0, hist1, hist2]  = HW2_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs); %trD
            [spatialTstD, hist0, hist1, hist2] = HW2_BoW.cmpFeatVecs(tstIds, scales, normH, normW, bowCs); %tstD
            
            %save('spatialTrD.mat','spatialTrD');
            save('spatialTstD.mat','spatialTstD');

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Write code for training logistic regression classifier and prediction here %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end;
                
        function bowCs = learnDictionary(scales, normH, normW)
            % Number of random patches to build a visual dictionary
            % Should be around 1 million for a robust result
            % We set to a small number her to speed up the process. 
            nPatch2Sample = 100000;
            
            % load train ids
            trIds = ml_load('../bigbangtheory/train.mat', 'imIds'); 
            nPatchPerImScale = ceil(nPatch2Sample/length(trIds)/length(scales));
                        
            randWins = cell(length(scales), length(trIds)); % to store random patches
            for i=1:length(trIds);
                ml_progressBar(i, length(trIds), 'Randomly sample image patches');
                im = imread(sprintf('../bigbangtheory/%06d.jpg', trIds(i)));
                im = double(rgb2gray(im));  
                for j=1:length(scales)
                    scale = scales(j);
                    winSz = [scale, scale];
                    stepSz = winSz/2; % stepSz is set to half the window size here. 
                    
                    % ML_SlideWin is a class for efficient sliding window 
                    swObj = ML_SlideWin(im, winSz, stepSz);
                    
                    % Randomly sample some patches
                    randWins_ji = swObj.getRandomSamples(nPatchPerImScale);
                    
                    % resize all the patches to have a standard size
                    randWins_ji = reshape(randWins_ji, [scale, scale, size(randWins_ji,2)]);                    
                    randWins{j,i} = imresize(randWins_ji, [normH, normW]);
                end
            end;
            randWins = cat(3, randWins{:});
            randWins = reshape(randWins, [normH*normW, size(randWins,3)]);
                                    
            fprintf('Learn a visual dictionary using k-means\n');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % You will use vl_kmeans here                                %
            % to learn a visual vocabulary                               %
            % Input: randWinds contains your data points                 %
            % Output: bowCs: centroids from k-means, one column for each centroid  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %Part 3.2, Question 1 solution
            numClusters = 1000;
            [bowCs, ~] = vl_kmeans(randWins, numClusters);

            
        end;
                
        function [D, hist0, hist1, hist2] = cmpFeatVecs(imIds, scales, normH, normW, bowCs)
            n = length(imIds);
            D = cell(1, n);
            startT = tic;
            for i=1:n
                ml_progressBar(i, n, 'Computing feature vectors', startT);
                im = imread(sprintf('../bigbangtheory/%06d.jpg', imIds(i))); 
                
                %*********histograms for different levels************
                hist0 = (HW2_BoW.calculateSpatialPyramid(im, scales, normH, normW, bowCs, 1))/4;
                hist1 = (HW2_BoW.calculateSpatialPyramid(im, scales, normH, normW, bowCs, 4))/4;
                hist2 = (HW2_BoW.calculateSpatialPyramid(im, scales, normH, normW, bowCs, 16))/2;
                
                %********concatenating the histograms********
                feat = [hist0,hist1,hist2];       
                %feat = hist0;
                D{i} = feat(:);
            end;
            D = cat(2, D{:});
            D = double(D);
            D = D./repmat(sum(D,1), size(D,1),1);
        end        
        
        % bowCs: d*k matrix, with d = normH*normW, k: number of clusters
        % scales: sizes to densely extract the patches. 
        % normH, normW: normalized height and width of patches
        function bowIds = cmpBowIds(im, scales, normH, normW, bowCs)
            im = double(rgb2gray(im));
            bowIds = cell(length(scales),1);
            for j=1:length(scales)
                scale = scales(j);
                winSz = [scale, scale];
                stepSz = winSz/2; % stepSz is set to half the window size here.
                
                % ML_SlideWin is a class for efficient sliding window
                swObj = ML_SlideWin(im, winSz, stepSz);
                nBatch = swObj.getNBatch();
                
                for u=1:nBatch
                    wins = swObj.getBatch(u);
                    
                    % resize all the patches to have a standard size
                    wins = reshape(wins, [scale, scale, size(wins,2)]);                    
                    wins = imresize(wins, [normH, normW]);
                    wins = reshape(wins, [normH*normW, size(wins,3)]);
                    
                    % Get squared distance between windows and centroids
                    dist2 = ml_sqrDist(bowCs, wins); % dist2: k*n matrix, 
                    
                    % bowId: is the index of cluster closest to a patch
                    [~, bowIds{j,u}] = min(dist2, [], 1);                     
                end                
            end;
            bowIds = cat(2, bowIds{:});
        end;        
        
        %*********function to get patches for different levels********
     
        function histogra = calculateSpatialPyramid(im, scales, normH, normW, bowCs, nParts)
            
            histogra = single.empty(0);
            sqrtNParts = sqrt(nParts);
            [m, n, k] = size(im);
            %********keeping all the patches in C*********
            C = mat2cell(im, m/sqrtNParts*ones(sqrtNParts,1), n/sqrtNParts*ones(sqrtNParts,1), k);
            [x, y] = size(C);
            for i=1:x
                for j=1:y
                    bowIdsn = HW2_BoW.cmpBowIds(C{i,j}, scales, normH, normW, bowCs);                
                    featn = hist(bowIdsn, 1:size(bowCs,2));
                    histogra = [histogra,featn];
                end
            end
        end
    end    
end

