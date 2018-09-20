function pLoad = getTags(dataSetName)

if strcmpi(dataSetName, 'jaad')
    pLoad = getJaadTags;
elseif strcmpi(dataSetName,'kitti')
     pLoad = {'lbls',{'Pedestrians', 'Person_sitting','Cyclist'},...
        'ilbls',{'Misc'}};
   
elseif strcmpi(dataSetName,'cityperson')
    pLoad = {'lbls',{'pedestrians'},...
        'ilbls',{'people','riders','sitting_person','other'}};
else
    pLoad = {'lbls',{'person'},'ilbls',{'people'}};
end
end

function pLoad = getJaadTags()
maxPedIdx = 50;
maxPeople = 20;
counterPed = 7;
counterPeop = 2;
name{1,1}= 'pedestrian';
name{1,2}= 'ped';
name{1,3}= 'pedestrian_p1';
name{1,4}= 'ped_p1';
name{1,5}= 'pedestrian_p2';
name{1,6}= 'ped_p2';
namePeop{1,1} = 'people';
for i=1:maxPedIdx
    
    name{1,counterPed}=sprintf('ped%d',i);
    counterPed = counterPed + 1;
    name{1,counterPed}=sprintf('pedestrian%d',i);
    counterPed = counterPed + 1;
    
    if i <= maxPeople
        namePeop{1,counterPeop}= sprintf('people%d',i);
        counterPeop = counterPeop +1;
    end
    
    for j=1:2
        name{1,counterPed }=sprintf('ped%d_p%d',i,j);
        counterPed = counterPed + 1;
        name{1,counterPed }=sprintf('pedestrian%d_p%d',i,j);
        counterPed = counterPed + 1;
        if i <= maxPeople
            namePeop{1,counterPeop}= sprintf('people%d_p%d',i,j);
            counterPeop = counterPeop +1;
        end
    end
end

pLoad={'lbls',name,'ilbls',namePeop};




end
