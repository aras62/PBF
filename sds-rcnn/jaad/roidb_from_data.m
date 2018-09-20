function roidb = roidb_from_data(imdb, flip, dataInfo, cache_dir, hRng, vRng, ignOcc, squarify)
% roidb = roidb_from_caltech(imdb, flip)
%   Package the roi annotations into the imdb.
%
%   Inspired by Ross Girshick's imdb and roidb code.
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------


roidb.name = imdb.name;
anno_path = imdb.annotDir;

% addpath(genpath('./external/code3.2.1'));

pLoad = getTags(dataInfo.dataSet);
pLoad = [pLoad 'hRng',[hRng inf],'vRng',vRng, 'ignOcc',ignOcc, 'squarify', squarify];

if flip
    cache_file = [cache_dir '/roidb_' imdb.name '_flip'];
else
    cache_file = [cache_dir '/roidb_' imdb.name];
end

cache_file = [cache_file, '.mat'];
if dataInfo.regenRoiFile && exist(cache_file, 'file')
    delete(cache_file);
end

try
    load(cache_file);
catch
    fprintf('Loading region proposals...');
    regions = [];
    
    fprintf('done\n');
    if isempty(regions)
        fprintf('Warrning: no windows proposal is loaded !\n');
        regions.boxes = cell(length(imdb.image_ids), 1);
    end
    
    height = imdb.sizes(1,1);
    width = imdb.sizes(1,2);
    files=bbGt('getFiles',{anno_path});
    
    num_gts = 0;
    num_gt_no_ignores = 0;
    for i = 1:length(files)
        tic_toc_print('%d / %d\n', i, length(files));
        [~,gts]=bbGt_jaad('bbLoad',files{i},pLoad);
        ignores = gts(:,end);
        num_gts  = num_gts + length(ignores);
        num_gt_no_ignores  = num_gt_no_ignores + (length(ignores)-sum(ignores));
        
        if flip
            % for ori
            x1 = gts(:,1);
            y1 = gts(:,2);
            x2 = gts(:,1) + gts(:,3);
            y2 = gts(:,2) + gts(:,4);
            gt_boxes = [x1 y1 x2 y2];
            roidb.rois(i*2-1) = attach_proposals(regions.boxes{i}, gt_boxes, ignores);
            
            % for flip
            x1_flip = width - gts(:,1) - gts(:,3);
            y1_flip = y1;
            x2_flip = width - gts(:,1);
            y2_flip = y2;
            gt_boxes_flip = [x1_flip y1_flip x2_flip y2_flip];
            roidb.rois(i*2) = attach_proposals(regions.boxes{i}, gt_boxes_flip, ignores);
            
            if 0
                % debugging visualizations
                im = imread(imdb.image_at(i*2-1));
                t_boxes = roidb.rois(i*2-1).boxes;
                for k = 1:size(t_boxes, 1)
                    showboxes2(im, t_boxes(k,1:4));
                    title(sprintf('%s, ignore: %d\n', imdb.image_ids{i*2-1}, roidb.rois(i*2-1).ignores(k)));
                    pause;
                end
                im = imread(imdb.image_at(i*2));
                t_boxes = roidb.rois(i*2).boxes;
                for k = 1:size(t_boxes, 1)
                    showboxes2(im, t_boxes(k,1:4));
                    title(sprintf('%s, ignore: %d\n', imdb.image_ids{i*2}, roidb.rois(i*2).ignores(k)));
                    pause;
                end
            end
        else
            % for ori
            x1 = gts(:,1);
            y1 = gts(:,2);
            x2 = gts(:,1) + gts(:,3);
            y2 = gts(:,2) + gts(:,4);
            gt_boxes = [x1 y1 x2 y2];
            roidb.rois(i) = attach_proposals(regions.boxes{i}, gt_boxes, ignores);
            if 0
                % debugging visualizations
                imdb.image_at(i);
                im = imread(imdb.image_at(i));
                t_boxes = gt_boxes; %roidb.rois(i).boxes;
                for k = 1:size(t_boxes, 1)
                    if ~gts(k,5)
                    showboxes2(im, t_boxes(k,1:4));
%                     title(sprintf('%s, ignore: %d\n', imdb.image_ids{i}, roidb.rois(i).ignores(k)));
                    pause;
                    end
                end
            end
            
        end
    end
    
    
    fprintf('Saving roidb to cache...\n');
    save(cache_file, 'roidb', '-v7.3');
    fprintf('done\n');
    
    fprintf('num_gt / num_ignore %d / %d \n', num_gt_no_ignores, num_gts);
end

end
% ------------------------------------------------------------------------
function rec = attach_proposals(boxes, gt_boxes, ignores)
% ------------------------------------------------------------------------

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]


all_boxes = cat(1, gt_boxes, boxes);
gt_classes = ones(size(gt_boxes, 1), 1); % set pedestrian label as 1
num_gt_boxes = size(gt_boxes, 1);

num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, 1, 'single');
for i = 1:num_gt_boxes
    rec.overlap(:, gt_classes(i)) = ...
        max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));

rec.ignores = ignores;
end
function showboxes2(im, boxes)


imshow(im);
hold on;
% Then, from the help:
rectangle('Position',[boxes(1),boxes(2),boxes(3)-boxes(1),boxes(4)-boxes(2)],...
        'LineWidth',2,'EdgeColor','b')



end