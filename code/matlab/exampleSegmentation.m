
img = imread('helamicroscopy.png');


%let's resize it a bit to speed it up
img_re = imresize(img,0.5);
img_s = img_re;
%img_s = img_re(1:1039,1:1387);
%%
sigma = 2;%1.5;
img_sfilt = imgaussfilt(img_s,sigma);

figure;
imshow(img_sfilt)

%%

tiledim = 30;

lambda = 5; minSizeMSER = 30; maxSizeMSER = 4000; maxVariation = 1;

maxEcc = .7; minSizeSplit = 30; maxSizeSplit = 1000;


%run the code (don't be scared it takes a bit)
bw = segmentImage(img_sfilt,'visualize',true);

%% rescale it back
bw_back = imresize(bw,2);
%dilate a bit the foreground such that closely neighboured cells appear in
%one cluster
figure;
imagesc(bw_back);colormap gray;