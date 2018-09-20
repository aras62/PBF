function dbExtract_jaad(tDir, flatten, skip,dataInfo, scale )
% Extract database to directory of images and ground truth text files.
%
% Call 'dbInfo(name)' first to specify the dataset. The format of the
% ground truth text files is the format defined and used in bbGt.m.
%
% USAGE
%  dbExtract( tDir, flatten )
%
% INPUTS
%  tDir     - [] target dir for image data (defaults to dataset dir)
%  flatten  - [0] if true output all images to single directory
%  skip     - [] specify frames to extract (defaults to skip in dbInfo)
%
% OUTPUTS
%
% EXAMPLE
%  dbInfo('InriaTest'); dbExtract;
%
% See also dbInfo, bbGt, vbb
%
% Caltech Pedestrian Dataset     Version 3.2.1
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

[pth,setIds,vidIds, ~, ext] = dbInfo_jaad;
if(nargin<1 || isempty(tDir)), tDir=pth; end
if(nargin<2 || isempty(flatten)), flatten=0; end
if(nargin<3 || isempty(skip)), [~,~,~,skip]=dbInfo; end
if(nargin<4 || isempty(dataInfo)), dataset = 'caltech'; else dataset = dataInfo.dataSet;end
if(nargin<5), scale = 1; end

if strcmpi(dataset, 'citypersons')
    extractCityPersons(tDir,  skip, scale,dataInfo, pth, vidIds, ext)
elseif strcmpi(dataset,'kitti')
    extractKitti(tDir,  skip, scale,dataInfo,pth, vidIds, ext)    
else
    extractSeqAndVbbDatasets(tDir, flatten, skip,dataset, scale,dataInfo,pth, setIds, vidIds);
end
end

function extractSeqAndVbbDatasets(tDir, flatten, skip,dataset, scale,dataInfo,pth, setIds, vidIds)
fileName = 'V%03d';
annotation_ext  = '/annotations';
if  exist([tDir '/images/'],'dir')
    genData = 0;
    fprintf('Images exist, only extracting %s annotations \n',dataInfo.anotationType);
else genData = 1;
end

if strcmp(dataset,'jaad')
    fileName = 'video_%04d';
    annotation_ext  = [annotation_ext '_' dataInfo.anotationType];
end

isSpp = strfind(tDir,'spp');
for s=1:length(setIds)
    for v=1:length(vidIds{s})
        % load ground truth
        name=sprintf(['set%02d/' fileName],setIds(s),vidIds{s}(v));
        if strcmpi(dataset,'jaad')
            A=vbb('vbbLoad',[pth '/annotations/vbb_' dataInfo.anotationType '/' name]);
        else
            A=vbb('vbbLoad',[pth '/annotations/' name]);
        end
        n=A.nFrame;
        fprintf('video_%04d.Total Number of Frames %d.\n',vidIds{s}(v), A.nFrame);
        if scale ~= 1,A = rescaleBBs(A,scale);end
        
        if(flatten), post=''; else post=[name '/']; end
        if(flatten), f=[name '_']; f(6)='_'; else f=''; end
        fs=cell(1,n); for i=1:n, fs{i}=[f 'I' int2str2(i-1,5)]; end
        if genData
            % extract images
            td=[tDir '/images/' post]; if(~exist(td,'dir')), mkdir(td); end
            if ~isempty(isSpp),  td1=[tDir '/images1/' post]; if(~exist(td1,'dir')), mkdir(td1); end; end;
            sr=seqIo([pth '/videos/' name '.seq'],'reader'); info=sr.getinfo();
            if ~isempty(isSpp),imageExtractSpp(fs,n,skip,sr,info,td,td1); else
                for i=skip-1:skip:n-1
                    f=[td fs{i+1} '.' info.ext];
                    if(exist(f,'file')), continue; end
                    sr.seek(i);I=sr.getframeb();f=fopen(f,'w');fwrite(f,I);fclose(f);
                end
            end;
            sr.close();
        end
        % extract ground truth
        td=[tDir  annotation_ext '/' post];
        for i=1:n, fs{i}=[fs{i} '.txt']; end
        if ~isempty(isSpp)
            vbb('vbbToFiles',A,td,fs,skip,2);
        else
            vbb('vbbToFiles',A,td,fs,skip,skip);
        end
    end
end

if scale ~= 1 && genData
    disp('Rescaling the images')
    rescaleImages(tDir,scale,isSpp, dataInfo.debugRescale);
end

end
function extractCityPersons(tDir, skip, scale,dataInfo,pth, vidIds, ext)
img_dir = [tDir '/images/'];
if  exist(img_dir,'dir')
    genData = 0;
    fprintf('Images exist, only extracting %s annotations \n',dataInfo.anotationType);
else
    genData = 1;
    mkdir(img_dir);
end

data_dirs = bbGt('getFiles', {fullfile(pth, vidIds{1})}, 1);
if genData == 1
    for i = 1: length(data_dirs)
        [images_fp, images_nms] = bbGt('getFiles', data_dirs(i), 1);
        
        for j = skip - 1: skip :length(images_fp) - 1
            if scale == 1
                copyfile(images_fp{j+1}, img_dir);
            else
                I = imread(images_fp{j+1});
                I = imresize(I,scale, 'bilinear');
                imwrite(I, fullfile(img_dir,[images_nms{j+1} '.' ext]));
            end
        end
    end
end

if length(vidIds) == 1
    disp('No annotations available for test data')
else
    if ~exist(fullfile(tDir,'annotations'), 'dir'), mkdir(fullfile(tDir,'annotations')); end;
    annots = load(fullfile(pth,'annotations',vidIds{2}));
    label_to_class = containers.Map({0,1,2,3,4,5}, {'ignore', 'pedestrians', 'riders', 'person_sitting', 'other', 'people'});
    fn = fieldnames(annots);
    annots = getfield(annots, fn{1});
    index = 1;
    cityName = annots{1}.cityname;
    city_annot = {{}};
    % Convert
    for i = 1: length(annots)
        if strcmpi(annots{i}.cityname, cityName)
            city_annot{index}{end+1} = {annots{i}.im_name,annots{i}.bbs};
        else
            cityName = annots{i}.cityname;
            index = index + 1;
            city_annot{index} = {};
            city_annot{index}{end+1} = {annots{i}.im_name,annots{i}.bbs};
        end
    end
    cnt = 0;
    for c = 1: length(city_annot)
        for i = skip - 1: skip : length(city_annot{c})-1
            [~,fname] = fileparts(city_annot{c}{i+1}{1});
            bbox = city_annot{c}{i+1}{2};
            fid = fopen(fullfile(tDir,'annotations',[fname '.txt']), 'w');
            fprintf(fid, '%% bbGt version=3\n');
            for j = 1: size(bbox,1)
                % Convert from class_label, x1,y1,w,h, instance_id, x1_vis, y1_vis, w_vis, h_vis
                % to label x1 y1 w h occ vx1 vy1 vw vh ign ang
                fprintf(fid, '%s %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n', ...
                    label_to_class(bbox(j,1)), bbox(j,2:5)*scale, (bbox(j,9) && bbox(j,10)), bbox(j,7:10)*scale, bbox(j,1) == 0, 0);
            end
            fclose(fid);
            cnt  = cnt +1;
        end
    end
end
end
function extractKitti(tDir, skip, scale,dataInfo,pth, vidIds, ext)

img_dir = [tDir '/images/'];
if  exist(img_dir,'dir')
    genData = 0;
    fprintf('Images exist, only extracting %s annotations \n',dataInfo.anotationType);
else
    mkdir(img_dir);
    if ~isempty(vidIds), genData = 1; else genData = 0; end
end

if genData == 1
    [images_fp, images_nms] = bbGt('getFiles', {fullfile(pth, vidIds{1})}, 1);
    for j = skip - 1: skip :length(images_fp) - 1
        if scale == 1
            copyfile(images_fp{j+1}, img_dir);
        else
            I = imread(images_fp{j+1});
            I = imresize(I,scale, 'bilinear');
            imwrite(I, fullfile(img_dir,[images_nms{j+1} '.' ext]));
        end
    end
end

if length(vidIds) == 1
    disp('No annotations available for test data')
else
    if ~exist(fullfile(tDir,'annotations'), 'dir'), mkdir(fullfile(tDir,'annotations')); end;
    if isempty(vidIds), return; end
    [files_pth, files_nms] = bbGt('getFiles', {fullfile(pth, vidIds{2})}, 1);
    for i = skip - 1: skip: length(files_pth)-1
        fid = fopen(files_pth{i+1},'r');
        fields  = textscan(fid,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' ');
        class_labels = fields{1};
        trun = fields{2};
        occ = fields{3};
        bbox = [fields{5},fields{6},fields{7}-fields{5},fields{8}-fields{6}];
        fclose(fid);
        fid = fopen(fullfile(tDir,'annotations',[files_nms{i+1} '.txt']),'w');
        fprintf(fid, '%% bbGt version=3\n');
        for j = 1:length(class_labels)
            if strcmpi(class_labels{j},'DontCare') || trun(j) > dataInfo.truncation
                ign = 1;
            else
                ign = 0;
            end;
            fprintf(fid, '%s %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n', ...
                class_labels{j},  bbox(j,:)*scale, occ(j) , [0,0,0,0], ign, 0);
        end
        fclose(fid);
    end
    
end
end

function imageExtractSpp(fs,n,skip,sr,info,td,td1)
if mode(n,2)
    numFrames = n-1;
else
    numFrames = n;
end
for i=1:skip:numFrames
    f=[td fs{i} '.' info.ext]; if(exist(f,'file')), continue; end
    sr.seek(i-1);I=sr.getframeb();f=fopen(f,'w');fwrite(f,I);fclose(f);
    f=[td1 fs{i+1} '.' info.ext]; if(exist(f,'file')), continue; end
    sr.seek(i);I=sr.getframeb();f=fopen(f,'w');fwrite(f,I);fclose(f);
end
end
function rescaleImages(tDir,scale,isSpp,dbrescale)
dirs1 = bbGt('getFiles', {[tDir '/images/']}, 1);
parfor i = 1:length(dirs1)
    if dbrescale, disp(dirs1{i}); end;
    I = imread(dirs1{i});
    I = imresize(I,scale, 'bilinear');
    imwrite(I, dirs1{i});
end
if ~isempty(isSpp)
    dirs1 = bbGt('getFiles', {[tDir '/images1/']}, 1);
    parfor i = 1:length(dirs1)
        if dbrescale, disp(dirs1{i}); end;
        I = imread(dirs1{i});
        I = imresize(I,scale, 'bilinear');
        imwrite(I, dirs1{i});
    end
end

end
function A = rescaleBBs(A,scale)
for i=1:A.nFrame
    for j=1:length(A.objLists{i})
        if ~isempty(A.objLists{i}(j).pos)
            A.objLists{i}(j).pos = floor(A.objLists{i}(j).pos*scale);
            A.objLists{i}(j).posv = floor(A.objLists{i}(j).posv*scale);
            
        end
    end
end
end
