function  [mr, recall] = evaluate_results_rpn_val(conf, net, output_dir, val_dir,valTest)


if (exist(output_dir, 'dir')), rmdir(output_dir, 's'); end

[imgNms,imgId] = bbGt('getFiles',{val_dir});

tic;

for imind=1:length(imgNms)
    
    im = imread(imgNms{imind});
    
    if strcmp(conf.dataInfo.dataSet, 'jaad')
        reg = regexp(imgId{imind}, '(set\d\d)_(video_\d\d\d\d)_I(\d\d\d\d\d)', 'tokens');
    else
        reg = regexp(imgId{imind}, '(set\d\d)_(V\d\d\d)_I(\d\d\d\d\d)', 'tokens');
    end
    
    setname = reg{1}{1};
    vname = reg{1}{2};
    iname = reg{1}{3};
    inum = str2num(iname) + 1;
    
    mkdir_if_missing(output_dir);
    fid  = fopen([output_dir '/' setname '_' vname '.txt'], 'a');
    
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
            fprintf(fid, '%d,%.3f,%.3f,%.3f,%.3f,%.3f\n', [inum x1 y1 w h score]);
        end
    end
    
    dt = toc/imind;
    
end

fclose(fid);

if strcmpi(valTest,'val')
    gt_path =  conf.dataInfo.valAnnotDir;
else
    gt_path =  conf.dataInfo.testAnnotDir;
    
end


[mr, ~, recall] = evaluate_result_dir_val({output_dir}, conf.test_min_h,gt_path,conf.dataInfo.dataSet);

end