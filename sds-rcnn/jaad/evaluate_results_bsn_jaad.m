function  evaluate_results_bsn_jaad(rpn_prototxt, rpn_weights, rpn_conf,...
    rcnn_prototxt, rcnn_weights, rcnn_conf, bbsNm, imlist)
rpn_net = caffe.Net(rpn_prototxt, 'test');
rpn_net.copy_from(rpn_weights);

rcnn_net = caffe.Net(rcnn_prototxt, 'test');
rcnn_net.copy_from(rcnn_weights);

results = [];
for imind=1:length(imlist)
    disp(imlist{imind})
    
    im = imread(imlist{imind});
    [boxes, scores, feat_scores_bg, feat_scores_fg] = proposal_im_detect(rpn_conf, rpn_net, im);
    
    % filter rpn
    proposal_num = rcnn_conf.test_batch_size;
    [aboxes, inds] = nms_filter([boxes, scores], rpn_conf.nms_per_nms_topN, rpn_conf.nms_overlap_thres, proposal_num, 1);
    
    boxes = aboxes(:, 1:4);
    scores = aboxes(:, 5);
    
    feat_scores_bg = feat_scores_bg(inds,:);
    feat_scores_fg = feat_scores_fg(inds,:);
    
    feat_scores_bg = feat_scores_bg(1:min(length(aboxes), proposal_num), :);
    feat_scores_fg = feat_scores_fg(1:min(length(aboxes), proposal_num), :);
    
    impadH = round(size(im,1)*rcnn_conf.padfactor);
    impadW = round(size(im,2)*rcnn_conf.padfactor);
    im = padarray(im, [impadH impadW]);
    
    rois_batch = single(zeros([rcnn_conf.crop_size(2) rcnn_conf.crop_size(1) 3 proposal_num]));
    
    for j=1:proposal_num
        
        % get box info
        x1 = boxes(j, 1); y1 = boxes(j, 2);
        x2 = boxes(j, 3); y2 = boxes(j, 4);
        w = x2-x1; h = y2-y1;
        
        x1 = x1 - w*rcnn_conf.padfactor + impadW;
        y1 = y1 - h*rcnn_conf.padfactor + impadH;
        w = w + w*rcnn_conf.padfactor;
        h = h + h*rcnn_conf.padfactor;
        
        % crop and resize proposal
        propim = imcrop(im, [x1 y1 w h]);
        propim = imresize(single(propim), [rcnn_conf.crop_size(1) rcnn_conf.crop_size(2)]);
        propim = bsxfun(@minus, single(propim), rcnn_conf.image_means);
        
        % permute data into caffe c++ memory, thus [num, channels, height, width]
        propim = propim(:, :, [3, 2, 1], :);
        propim = permute(propim, [2, 1, 3, 4]);
        rois_batch(:,:,:,j) = single(propim);
        
    end
    
    net_inputs = {rois_batch};
    rois_feat_scores = [feat_scores_bg feat_scores_fg];
    rois_feat_scores = single(rois_feat_scores(1:proposal_num, :));
    rois_feat_scores = single(permute(rois_feat_scores, [3, 4, 2, 1]));
    net_inputs{length(net_inputs) + 1} = rois_feat_scores;
    
    rcnn_net = reshape_input_data(rcnn_net, net_inputs);
    rcnn_net.forward(net_inputs);
    
    
    cls_scores_fused = rcnn_net.blobs(rcnn_conf.cls_score_sm).get_data();
    cls_scores_fused = cls_scores_fused(end,:);
    cls_scores_fused = cls_scores_fused(:);
    
    % score 1 (fused)
    aboxes = [boxes, cls_scores_fused];
    
    for scoreind=1:size(aboxes,1)
        
        x1 = aboxes(scoreind, 1);
        y1 = aboxes(scoreind, 2);
        x2 = aboxes(scoreind, 3);
        y2 = aboxes(scoreind, 4);
        score = aboxes(scoreind, 5);
        w = x2 - x1; h = y2 - y1;
        results = cat(1,results,[imind x1 y1 w h score]);
        
    end
    
    d=fileparts(bbsNm); if(~isempty(d)&&~exist(d,'dir')), mkdir(d);end;
    dlmwrite(bbsNm,results);
    
end

