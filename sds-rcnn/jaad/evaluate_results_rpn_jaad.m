function  evaluate_results_rpn_jaad(rpn_prototxt, rpn_weights,conf, bbsNm, imlist)

results = [];
net = caffe.Net(rpn_prototxt, 'test');
net.copy_from(rpn_weights);
for imind=1:length(imlist)
    
    disp(imlist{imind})
    im = imread(imlist{imind});
    [pred_boxes, scores] = proposal_im_detect(conf, net, im);
    aboxes = [pred_boxes scores];
    
    [aboxes, valid] = nms_filter(aboxes, conf.nms_per_nms_topN, conf.nms_overlap_thres, conf.nms_after_nms_topN , true);

    for boxind=1:size(aboxes,1)
        x1 = (aboxes(boxind, 1));
        y1 = (aboxes(boxind, 2));
        x2 = (aboxes(boxind, 3));
        y2 = (aboxes(boxind, 4));
        score = aboxes(boxind, 5);
        w = x2 - x1 + 1;
        h = y2 - y1 + 1;
        if score >= 0.001
             results = cat(1,results,[imind x1 y1 w h score]);
        end
    end
    
%         imshow(im);
%         hold on;
%         for boxind=1:size(aboxes,1)
%             if aboxes(boxind, 5) >= 0.001
%             rectangle('Position',[aboxes(boxind,1),aboxes(boxind,2),aboxes(boxind,3)-aboxes(boxind,1),aboxes(boxind,4)-aboxes(boxind,2)],...
%                 'LineWidth',2,'EdgeColor','b')
%             aboxes(boxind,5)
%             pause;
%             end
%         end
    
end


d=fileparts(bbsNm); if(~isempty(d)&&~exist(d,'dir')), mkdir(d);end;
dlmwrite(bbsNm,results);

end