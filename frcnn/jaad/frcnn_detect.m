function bbsNm = frcnn_detect(dataInfo, modelSetup)
close all;
clc;

dataset = dataInfo.dataSet;
testDir = dataInfo.testDir ;
testImgsRange = dataInfo.testImgsRange;
reapply = modelSetup.reapply;
resultsPath = dataInfo.resultsPath;
modelName = modelSetup.modelName;


scalestr = num2str(dataInfo.scale);
dotIdx = strfind(scalestr,'.');
if ~isempty(dotIdx), scalestr(dotIdx)=''; end;

if ~isempty(testImgsRange)
    bbsNm=[resultsPath 'frcnn/' modelName '_' modelSetup.networkType '/'  dataset '/' num2str(testImgsRange(1)) '-' num2str(testImgsRange(2)) '_s' scalestr '_Dets.txt'];
else bbsNm=[resultsPath 'frcnn/' modelName '_' modelSetup.networkType '/'  dataset '/all' '_s' scalestr '_Dets.txt']; end

if(reapply && exist(bbsNm,'file')), delete(bbsNm); end
if(~reapply && exist(bbsNm,'file')), fprintf('the detection file %s already exists. \n',bbsNm); return;end;


% caffe.reset_all();


%% -------------------- INIT_MODEL --------------------
if strcmpi(modelSetup.networkType,'vgg')
    model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', [modelSetup.modelName '_vgg']); %% VGG-16
elseif strcmpi(modelSetup.networkType,'zf')
    model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', [modelSetup.modelName '_zf']); %% ZF
else fprintf('Model %s does not exist, select either vgg or zf.\n',modelSetup.networkType); bbsNm = []; return; end;
imgNms = bbGt('getFiles', {testDir});
if ~isempty(testImgsRange), imgNms = imgNms(testImgsRange(1):testImgsRange(2));end

%% -------------------- CONFIG --------------------
opts.per_nms_topN           = modelSetup.per_nms_topN;
opts.nms_overlap_thres      = modelSetup.nms_overlap_thres;
opts.after_nms_topN         = modelSetup.after_nms_topN;
opts.use_gpu                = modelSetup.useGpu;
opts.test_scales            = modelSetup.test_scales;
opts.test_min_box_height    = modelSetup.hRng(1,1);


%% -------------------- INIT_MODEL --------------------

proposal_detection_model    = load_proposal_detection_model(model_dir,modelSetup.rpnOnly);
proposal_detection_model.conf_proposal.test_max_size = modelSetup.test_max_size;
proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_proposal.test_min_box_height = opts.test_min_box_height;

if opts.use_gpu
    proposal_detection_model.conf_proposal.image_means = gpuArray(proposal_detection_model.conf_proposal.image_means);
end

% proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);



% set gpu/cpu
if opts.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end

%% -------------------- TESTING --------------------
results = [];

if modelSetup.rpnOnly
    
    for j = 1:length(imgNms)
        
        im = imread(imgNms{j});
        fprintf('%s\n',imgNms{j});
        
        if opts.use_gpu
            im = gpuArray(im);
        end
        [boxes, scores]   = proposal_im_detect_jaad(proposal_detection_model.conf_proposal, rpn_net, im);
        thres = modelSetup.detectionThresh;%0.001;
        abox = [boxes, scores];
        abox = boxes_filter(abox, opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
        I = abox(:, 5) >= thres;
        dets = abox(I, :);
        
        % orignal bbs are two cordinates. convert to rectangle coordinates
        for boxind=1:size(dets,1)
            x1 = dets(boxind,1);
            y1 = dets(boxind,2);
            x2 = dets(boxind,3);
            y2 = dets(boxind,4);
            score = dets(boxind,5);
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            results = cat(1,results,[j x1 y1 w h score]);
        end
        
        if modelSetup.debug
            imshow(im);
            hold on;
            for boxind=1:size(dets,1)
                rectangle('Position',[dets(boxind,1),dets(boxind,2),dets(boxind,3)-dets(boxind,1),dets(boxind,4)-dets(boxind,2)],...
                    'LineWidth',2,'EdgeColor','b')
                dets(boxind,5)
                pause;
            end
        end
    end
    
else
    % fast rcnn net
    fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
    fast_rcnn_net.copy_from(proposal_detection_model.detection_net);
    proposal_detection_model.conf_detection.test_scales = opts.test_scales;
    proposal_detection_model.conf_detection.fcnn_cls_score = modelSetup.fcnn_cls_score;
    if opts.use_gpu
        proposal_detection_model.conf_detection.image_means = gpuArray(proposal_detection_model.conf_detection.image_means);
    end
    
    for j = 1:length(imgNms)
        
        disp(imgNms{j})
        im = imread(imgNms{j});
        if opts.use_gpu
            im = gpuArray(im);
        end
        [boxes, scores]             = proposal_im_detect_jaad(proposal_detection_model.conf_proposal, rpn_net, im);
        aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
        
        if proposal_detection_model.is_share_feature
            [boxes, scores]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
                rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
                aboxes(:, 1:4), opts.after_nms_topN);
        else
            [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
                aboxes(:, 1:4), opts.after_nms_topN);
        end
        
        % visualize
        %% for pedestrians (for VOC)
        
        if modelSetup.detVOC
            thres = modelSetup.detectionThresh;
            boxes_cell{15} = [boxes(:, (1+(15-1)*4):(15*4)), scores(:, 15)];
            boxes_cell{15} = boxes_cell{15}(nms(boxes_cell{15}, 0.3), :);
            I = boxes_cell{15}(:, 5) >= thres;
            dets = boxes_cell{15}(I, :);
        else
            thres = modelSetup.detectionThresh;
            res =  [boxes, scores] ;
            I =  res(:, 5) >= thres;
            dets =  res(I, :);
        end  
            % orignal bbs are two cordinates. convert to rectangle coordinates
            for boxind=1:size(dets,1)
                x1 = dets(boxind,1);
                y1 = dets(boxind,2);
                x2 = dets(boxind,3);
                y2 = dets(boxind,4);
                score = dets(boxind,5);
                w = x2 - x1 + 1;
                h = y2 - y1 + 1;
                results = cat(1,results,[j x1 y1 w h score]);
                
            end
            if modelSetup.debug
                imshow(im);
                hold on;
                for boxind=1:size(dets,1)
                    rectangle('Position',[dets(boxind,1),dets(boxind,2),dets(boxind,3)-dets(boxind,1),dets(boxind,4)-dets(boxind,2)],...
                        'LineWidth',2,'EdgeColor','b')
                    dets(boxind,5)
                    pause;
                end
            end
            
        end
    end
    d=fileparts(bbsNm); if(~isempty(d)&&~exist(d,'dir')), mkdir(d);end;
    dlmwrite(bbsNm,results);
    caffe.reset_all();
    % clear mex;
    
end

    function proposal_detection_model = load_proposal_detection_model(model_dir, do_det)
        ld                          = load(fullfile(model_dir, 'model'));
        proposal_detection_model    = ld.proposal_detection_model;
        clear ld;
        
        proposal_detection_model.proposal_net_def ...
            = fullfile(model_dir, proposal_detection_model.proposal_net_def);
        proposal_detection_model.proposal_net ...
            = fullfile(model_dir, proposal_detection_model.proposal_net);
        if ~do_det
            proposal_detection_model.detection_net_def ...
                = fullfile(model_dir, proposal_detection_model.detection_net_def);
            proposal_detection_model.detection_net ...
                = fullfile(model_dir, proposal_detection_model.detection_net);
        end
    end
    function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
        % to speed up nms
        if per_nms_topN > 0
            aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
        end
        % do nms
        if nms_overlap_thres > 0 && nms_overlap_thres < 1
            valid = nms_jaad(aboxes, nms_overlap_thres, use_gpu);
            aboxes = aboxes(valid, :);
        end
        if after_nms_topN > 0
            aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
        end
    end
    function pick = nms_jaad(boxes, overlap, use_gpu, inside_thres)
        % top = nms(boxes, overlap)
        % Non-maximum suppression. (FAST VERSION)
        % Greedily select high-scoring detections and skip detections
        % that are significantly covered by a previously selected
        % detection.
        %
        % NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
        % but an inner loop has been eliminated to significantly speed it
        % up in the case of a large number of boxes
        
        % Copyright (C) 2011-12 by Tomasz Malisiewicz
        % All rights reserved.
        %
        % This file is part of the Exemplar-SVM library and is made
        % available under the terms of the MIT license (see COPYING file).
        % Project homepage: https://github.com/quantombone/exemplarsvm
        
        
        if isempty(boxes)
            pick = [];
            return;
        end
        
        if ~exist('use_gpu', 'var')
            use_gpu = false;
        end
        
        if ~exist('inside_thres', 'var')
            inside_thres = 100000000000;
        end
        
        if use_gpu
            s = boxes(:, end);
            if ~issorted(s(end:-1:1))
                [~, I] = sort(s, 'descend');
                boxes = boxes(I, :);
                pick = nms_gpu_mex_jaad(single(boxes)', double(overlap), double(inside_thres));
                pick = I(pick);
            else
                pick = nms_gpu_mex_jaad(single(boxes)', double(overlap), double(inside_thres));
            end
            return;
        end
        
        if size(boxes, 1) < 1000000
            pick = nms_mex_jaad(double(boxes), double(overlap));
            return;
        end
        
        x1 = boxes(:,1);
        y1 = boxes(:,2);
        x2 = boxes(:,3);
        y2 = boxes(:,4);
        s = boxes(:,end);
        
        area = (x2-x1+1) .* (y2-y1+1);
        [vals, I] = sort(s);
        
        pick = s*0;
        counter = 1;
        while ~isempty(I)
            last = length(I);
            i = I(last);
            pick(counter) = i;
            counter = counter + 1;
            
            xx1 = max(x1(i), x1(I(1:last-1)));
            yy1 = max(y1(i), y1(I(1:last-1)));
            xx2 = min(x2(i), x2(I(1:last-1)));
            yy2 = min(y2(i), y2(I(1:last-1)));
            
            w = max(0.0, xx2-xx1+1);
            h = max(0.0, yy2-yy1+1);
            
            inter = w.*h;
            o = inter ./ (area(i) + area(I(1:last-1)) - inter);
            
            I = I(find(o<=overlap));
        end
        
        pick = pick(1:(counter-1));
    end