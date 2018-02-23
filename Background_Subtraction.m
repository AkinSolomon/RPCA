%% ECE 7866 Computer Vision
%  HW4
%  Akinlawon Solomon

%% Initialization
clear; close all; clc

%% ================Data Pre-Processing=================

imfile = dir('images4/');
imgs = {imfile(~[imfile.isdir]).name};
imsubset = 1:222;
p = length(imsubset);


for i = 1:length(imsubset)
    
    imstcolor(:,:,:,i) = imread(imgs{1,imsubset(i)});
    imtemp = double(rgb2gray(imstcolor(:,:,:,i)));
    imstgray(:,:,i) = (imtemp/255);
end

%Construct D matrix
D = [];
for i = 1:length(imsubset)
    
    imtemp = imstgray(:,:,i);
    D(:,i) = imtemp(:);
    
end
N = size(D,1);




%% ============  Step 1: Robust PCA ===================
[mu_rob,Brob,Crob,Wopt,evo] = rPCA(D,size(imtemp),55);

clear J large closed new_img 
disp('%==============================================================================%');
%% ============= Step 2: Weighted PCA ================

k = 25;
max_iter = 100;
iter = 1;
flag = 1;
Brobw = [Brob 0.0001*randn(N,k - size(Brob,2))];
Crobw = [Crob; 0.0001*randn(k-size(Crob,1),p)];
evo2=zeros(20,2);

while iter<max_iter && flag
    
    errpix = (D - mu_rob*ones(1,p) - Brobw*Crobw);
    errpixtot = sum(sum((errpix.^2).*Wopt));
    
    Ctemp = Crobw;
    Btemp = Brobw;
    
    mu_rob = sum(Wopt.*(D - Brobw*Crobw),2)./sum(Wopt,2);
    
    %weighted coeff
    for i = 1:size(Crobw,2)
        WBrob = Wopt(:,i)*ones(1,size(Brobw,2)).*Brobw;
        Crobw(:,i) = (Brobw'*WBrob)\WBrob'*(D(:,i)- mu_rob);
    end
    
    %weighted bases
    for i = 1:size(Brobw,1)
        Wcrob = Crobw.*(ones(size(Crobw,1),1)*Wopt(i,:));
        Brobw(i,:)=((Crobw*Wcrob')\Wcrob*(D(i,:)- mu_rob(i)*ones(1,p))')';
    end
    
    %angular error
    
    ang_err = subspace(Brobw,Btemp);
    
    if ang_err<1e-3 || (iter>15)
        flag = 0;
    end
    
       
    fprintf('Iter:%d , Err:%.3f ,  angular_error: %.3f \n',iter,errpixtot, ang_err);   
    evo2(iter,:) = [errpixtot ang_err];
    iter = iter + 1;
    
end

%% ==============Display Images=================
%Calculate Foreground Matrix S and Background Matrix L
L = mu_rob + Brobw*Crobw;
S = D - L;
np = [70:7:220]; %change np to see more images

for i=1:length(np)
J = (reshape(S(:,np(i)),size(imtemp)));
% figure;
% imagesc(J),colormap('gray')
new_img = imbinarize(J,0.1);
large = bwareaopen(new_img,20);
closed = bwmorph(large,'close');
figure;
imshow(closed)

end

hold off
% writerObj = VideoWriter('myVideo.avi');
% writerObj.FrameRate = 5;
% open(writerObj);
% 
% for i=1:length(imsubset)
%     J = imadjust(reshape(S(:,i),size(imtemp)));
%     new_img = imbinarize(((J)));
%     large = bwareaopen(new_img,20);
%     closed = bwmorph(large,'close');
%     map = colormap(gray(2));
%     frame = im2frame(uint8(J),map);
%     
%     writeVideo(writerObj,frame);
% end

