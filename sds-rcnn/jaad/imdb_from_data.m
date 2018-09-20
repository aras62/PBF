function imdb = imdb_from_data(dataInfo, image_set, flip, cache_dir)
% imdb = imdb_from_voc(root_dir, image_set, year)
%   Builds an image database for the PASCAL VOC devkit located
%   at root_dir using the image_set and year.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
%imdb.name = 'voc_train_2007'
%imdb.image_dir = '/work4/rbg/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
%imdb.extension = '.jpg'
%imdb.image_ids = {'000001', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'aeroplane', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest



dataSetName = dataInfo.dataSet;
cache_file = [cache_dir '/imdb_' dataSetName '_' image_set];
if flip
    cache_file = [cache_file, '_flip'];
end
cache_file = [cache_file, '.mat'];

if dataInfo.regenImdbFile && exist(cache_file, 'file')
    delete(cache_file);
end

try
    load(cache_file);
catch
    
    if strcmp(image_set,'test')
        root_dir = dataInfo.testDir;
        annot_dir = dataInfo.testAnnotDir;
    elseif strcmp(image_set,'train')
        root_dir = dataInfo.trainDir;
        annot_dir = dataInfo.trainAnnotDir;
    elseif  strcmp(image_set,'val')
        root_dir = dataInfo.valDir;
        annot_dir = dataInfo.valAnnotDir;
    end
    
    imdb.name = [dataSetName '_' image_set];
    [imgNms,imgIds] = bbGt('getFiles',{root_dir});
    imdb.image_dir = root_dir;
    imdb.annotDir = annot_dir;
    imdb.extension = imgNms{1}(end-3:end); %'jpg';
    imdb.flip = flip;
    imdb.image_ids = cell(length(imgNms), 1);
    imdb.image_path = imgNms;
    if flip
        for i = 1:length(imgNms)
            imdb.image_ids{i*2-1} = imgIds{i};
            imdb.image_ids{i*2} = [imgIds{i} '_flip'];
            if ~exist([imdb.image_dir imgIds{i} '_flip' imdb.extension], 'file')
                im = imread(imgNms{i});
                imwrite(fliplr(im), [imdb.image_dir imgIds{i} '_flip' imdb.extension]);
            end
        end
    else
        imdb.image_ids = imgIds;
    end
    
    imdb.image_at = @(i) ...
        fullfile(imdb.image_dir, [imdb.image_ids{i} imdb.extension]);
    
    imdb.classes{1} = 'pedestrian';
    imdb.num_classes = 1;
    
    %     if strcmpi(dataSetName, 'inria') || strcmpi(dataSetName, 'jaad')
    for i = 1:length(imgNms)
        try
            im = imread(imgNms{i});
        catch
            error('ERROR: failed to load image %s!\n', imgNms{i});
        end
        s = size(im);
        imdb.sizes(i, :) = s(1:2); % the size is fix for caltech
    end
    %     else
    %         im = imread(imgNms{1});
    %         s = size(im);
    %         imdb.sizes = repmat(s(1:2), [length(imgNms),1]);
    %     end
    
    imdb.roidb_func = @roidb_from_data;
    
    fprintf('Saving imdb to cache...');
    save(cache_file, 'imdb', '-v7.3');
    fprintf('done\n');
end
