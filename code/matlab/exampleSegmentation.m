%%
%arg1 = 0 for gaussian filter with sigma 2
%       1 for median filter with 5*5 window size
%       2 for median filter with 3*3 window size (default)
%%
%% TO NOTE: Not cropping the image boundaries! Do it when processing it later.'visualize',true);
function [] = exapmleSegmentation(arg1)
    if nargin == 0 
        arg1 = '2' %default 
    end
    display(arg1)
    direc = dir('directory_path_to_images');
    save_direc = 'directory_path_to_save_masks';
   
    for file = direc'
        if endsWith(file.name, '.png')
            fname = strcat(file.folder ,'/', file.name);
            img = imread(fname);

            %let's resize it a bit to speed it up
            img_re = imresize(img,0.5);
            img_s = img_re;

            %img_s = img_re(1:1039,1:1387);
            
            if arg1 == '0' 
                sigma = 2; %1.5;
                img_sfilt = imgaussfilt(img_s,sigma);
            elseif arg1 == '1'
                 img_med_filter = medfilt2(img_s, [5,5]);
                 %img_sfilt = histeq(img_med_filter)
                 %img_sfilt = adapthisteq(img_med_filter,'NumTiles',[12 12],'ClipLimit',0.003);
                img_sfilt = img_med_filter;
            elseif arg1 == '2' 
                img_med_filter = medfilt2(img_s, [3,3]);
                img_sfilt = img_med_filter;
            else
                error("No implementation found!");
            end

            %figure;
            %imshow(img_sfilt)

            tiledim = 30;

            lambda = 5; minSizeMSER = 30; maxSizeMSER = 4000; maxVariation = 1;

            maxEcc = .7; minSizeSplit = 30; maxSizeSplit = 1000;

            imgname = strsplit(file.name, '.');
            %run the code (don't be scared it takes a bit)
            bw = segmentImage(img_sfilt,'visualize',false);

            %% rescale it back
            bw_back = imresize(bw,2);
            %dilate a bit the foreground such that closely neighboured cells appear in
            %one cluster

            %se = strel('sphere',1); Tried dilating. Can use it while
            %checking with the lensfree images. Shows good result.
            %bw_back = imdilate(bw_back,se);
            fname = strcat(save_direc , imgname{1} , '.png');
            imwrite(bw_back, fname);
            
            %figure;
            %imagesc(bw_back);colormap gray;
        end
    end
end