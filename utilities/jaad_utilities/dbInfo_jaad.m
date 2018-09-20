function [pth,setIds,vidIds,skip,ext] = dbInfo_jaad( name1, dataPath1,jaadSubDatasetDir1,jaadSubDataset1 )
% Specifies data amount and location.
%
% 'name' specifies the name of the dataset. Valid options include: 'Usa',
% 'UsaTest', 'UsaTrain', 'InriaTrain', 'InriaTest', 'Japan', 'TudBrussels',
% 'ETH', and 'Daimler'. If dbInfo() is called without specifying the
% dataset name, defaults to the last used name (or 'UsaTest' on first call
% to dbInfo()). Finally, one can specify a subset of a dataset by appending
% digits to the end of the name (eg. 'UsaTest01' indicates first set of
% 'UsaTest' and 'UsaTest01005' indicate first set, fifth video).
%
% USAGE
%  [pth,setIds,vidIds,skip,ext] = dbInfo( [name] )
%
% INPUTS
%  name     - ['UsaTest'] specify dataset, caches last passed in name
%
% OUTPUTS
%  pth      - directory containing database
%  setIds   - integer ids of each set
%  vidIds   - [1xnSets] cell of vectors of integer ids of each video
%  skip     - specify subset of frames to use for evaluation
%  ext      - file extension determining image format ('jpg' or 'png')
%
% EXAMPLE
%  [pth,setIds,vidIds,skip,ext] = dbInfo
%
% See also
%
% Caltech Pedestrian Dataset     Version 3.2.1
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

persistent name dataPath jaadSubDatasetDir jaadSubDataset; % cache last used name
if(nargin && ~isempty(name1))
    name=lower(name1);
    dataPath = dataPath1;
    jaadSubDatasetDir = jaadSubDatasetDir1;
    jaadSubDataset = jaadSubDataset1;
else
    if(isempty(name))
        name='caltechtest';
    end; end;
name1=name;
dataPath1 = dataPath;
jaadSubDatasetDir1 = jaadSubDatasetDir;
jaadSubDataset1 = jaadSubDataset;
vidId=str2double(name1(end-2:end)); % check if name ends in 3 ints
if(isnan(vidId)), vidId=[]; else name1=name1(1:end-3); end
setId=str2double(name1(end-1:end)); % check if name ends in 2 ints
if(isnan(setId)), setId=[]; else name1=name1(1:end-2); end

switch name1
    case 'caltechtrain' % Caltech Pedestrian Datasets (training)
        setIds=0:5; subdir='Caltech'; skip=30; ext='jpg';
        vidIds={0:14 0:5 0:11 0:12 0:11 0:12};
    case 'caltechtest' % Caltech Pedestrian Datasets (testing)
        setIds=6:10; subdir='Caltech'; skip=30; ext='jpg';
        vidIds={0:18 0:11 0:10 0:11 0:11};
    case 'caltechval'
        setIds=7; subdir='Caltech'; skip=30; ext='jpg';
        vidIds={0:10};
        warning('Caltech does not have validation data. A subset of test data is used for validation')
    case 'inriatrain' % INRIA peds (training)
        setIds=0; subdir='INRIA'; skip=1; ext='png'; vidIds={0:1};
    case 'inriatest' % INRIA peds (testing)
        setIds=1; subdir='INRIA'; skip=1; ext='png'; vidIds={0};
    case 'inriaval' % INRIA peds (testing)
        setIds=1; subdir='INRIA'; skip=1; ext='png'; vidIds={0};
        warning('Inria does not have validation data. Test data is used for validation')
    case 'jaadtrain'
        setIds = 0; subdir='JAAD'; skip=10; ext='png';
        jaad_trainset = fullfile(jaadSubDatasetDir1, [jaadSubDataset1 '_trainSetIds.mat']);
        load(jaad_trainset)
    case 'jaadtest'
        setIds = 0; subdir='JAAD'; skip=10; ext='png'; 
        jaad_testset = fullfile(jaadSubDatasetDir1, [jaadSubDataset1 '_testSetIds.mat']);
        load(jaad_testset)
    case 'jaadval'
        setIds = 0; subdir='JAAD'; skip=10; ext='png'; 
        jaad_valset = fullfile(jaadSubDatasetDir1, [jaadSubDataset1 '_valSetIds.mat']);
        load(jaad_valset)
    case 'citypersontrain'
        setIds = 0; subdir='CityPerson'; skip=1; ext='png';
        vidIds = {'leftImg8bit/train', 'anno_train'};
    case 'citypersontest'
        setIds = 0; subdir='CityPerson'; skip=1; ext='png'; 
        vidIds = {'leftImg8bit/test'};
    case 'citypersonval'
        setIds = 0; subdir='CityPerson'; skip=1; ext='png'; 
        vidIds = {'leftImg8bit/val', 'anno_val'};
    case 'kittitest'
        setIds = 0; subdir='KITTI'; skip=1; ext='png'; 
        vidIds = {'data_object_image_2/testing/image_2'};
    case 'kittitrain'
        setIds = 0; subdir='KITTI'; skip=1; ext='png'; 
        vidIds = {'data_object_image_2/training/image_2', 'training/label_2'};
    case 'kittival'
        warning('KITTI does not have validation data!')
        setIds = 0; subdir='KITTI'; skip=1; ext='png'; 
        vidIds={};
    case 'tudbrussels' % TUD-Brussels dataset
        setIds=0; subdir='TudBrussels'; skip=1; ext='png'; vidIds={0};
    case 'eth' % ETH dataset
        setIds=0:2; subdir='ETH'; skip=1; ext='png'; vidIds={0 0 0};
    case 'daimler' % Daimler dataset
        setIds=0; subdir='Daimler'; skip=1; ext='png'; vidIds={0};
        
    otherwise, error('unknown data type: %s',name);
end

% optionally select only specific set/vid if name ended in ints
if(~isempty(setId)), setIds=setIds(setId); vidIds=vidIds(setId); end
if(~isempty(vidId)), vidIds={vidIds{1}(vidId)}; end
% actual directory where data is contained
pth=[dataPath1 subdir];

end
