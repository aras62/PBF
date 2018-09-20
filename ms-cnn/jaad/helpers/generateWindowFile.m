function generateWindowFile(dataInfo, modelSetup,dataType)
% Copyright (c) 2016 The Regents of the University of California
% see mscnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!

fprintf('Preparing window data for %s \n', dataType);
% set Caltech dataset directory and Piotr's toolbox
file_name = sprintf('ms-cnn/cache/mscnn_window_file_%s.txt',dataType);
if ~modelSetup.regenWindowFile && exist(file_name,'file'), return; end;

pLoad = getTags(dataInfo.dataSet);

pLoad = {'lbls', pLoad{2}};
igLoad ={'ilbls',pLoad{4}};

pLoad = [pLoad, 'hRng',modelSetup.hRng, 'vRng', ...
    modelSetup.vRng,'ignOcc',modelSetup.ignOcc, 'squarify',modelSetup.squarify];
if(~isempty(modelSetup.yRng)),  pLoad = [pLoad 'yRng', modelSetup.yRng]; end
if(~isempty(modelSetup.xRng)),  pLoad = [pLoad 'xRng', modelSetup.xRng]; end


% choose which data list to generate
if strcmp(dataType,'test')
    posImgDir= dataInfo.testDir;
    posGtDir = dataInfo.testAnnotDir;
elseif strcmp(dataType,'train')
    posImgDir= dataInfo.trainDir;
    posGtDir = dataInfo.trainAnnotDir;
elseif  strcmp(dataType,'val')
    posImgDir = dataInfo.valDir;
    posGtDir = dataInfo.valAnnotDir;
end
    

fs={posImgDir,posGtDir};
fs=bbGt('getFiles',fs); 
nImg=size(fs,2);

fid = fopen(file_name, 'wt');

show = modelSetup.showWindowData;
if (show)
  fig = figure(1); 
  set(fig,'Position',[-30 30 640 480]);
  hd.axes = axes('position',[0.1,0.1,0.8,0.8]);
end

for i = 1:nImg
  if (mod(i,500) == 0), fprintf('image idx: %i/%i\n', i, nImg); end
  img_path = fs{1,i};
  I=imread(img_path);
  if (show)
    imshow(I); axis(hd.axes,'image','off'); hold(hd.axes, 'on');
  end
  [imgH, imgW, channels]=size(I);
  [~,gt]=bbGt_jaad('bbLoad',fs{2,i},pLoad);
  [~,iggt]=bbGt('bbLoad',fs{2,i},igLoad);
  
  fprintf(fid, '# %d\n', i-1);
  fprintf(fid, '%s\n', img_path);
  fprintf(fid, '%d\n%d\n%d\n', channels, imgH, imgW);

  num_objs = size(gt,1);
  label = 1;
  fprintf(fid, '%d\n', num_objs);
  for j = 1:num_objs
    ignore = gt(j,5);
    w = gt(j,3); h = gt(j,4);
    w = max(w,2); h = max(h,2);
    x1 = gt(j,1); y1 = gt(j,2);
    x2 = x1+w-1; y2 = y1+h-1;
    fprintf(fid, '%d %d %d %d %d %d\n', label, ignore, round(x1), round(y1), round(x2), round(y2));

    if (show)
      if (ignore), color = 'g'; else color = 'r'; end
      rectangle('Position', [x1 y1 w h],'LineWidth',2,'edgecolor',color);   
      text(x1+0.5*w,y1,num2str(label),'color','r','BackgroundColor','k','HorizontalAlignment',...
         'center','VerticalAlignment','bottom','FontWeight','bold','FontSize',8);
    end
  end
  
  num_dontcare = size(iggt,1);
  fprintf(fid, '%d\n', num_dontcare);
  for j  = 1:num_dontcare
    w = iggt(j,3); h = iggt(j,4);
    w = max(w,2); h = max(h,2);
    x1 = iggt(j,1); y1 = iggt(j,2);
    x2 = x1+w-1; y2 = y1+h-1;
    fprintf(fid, '%d %d %d %d\n', round(x1), round(y1), round(x2), round(y2));
    if (show)
      rectangle('Position', [x1 y1 x2-x1 y2-y1],'LineWidth',2.5,'edgecolor','y');
    end
  end
  if (show), pause(0.01); end
end

fclose(fid);
end
