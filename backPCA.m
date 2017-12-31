function [L,S] = rPCA(Xdata,im_sz,percvar)

%collecting info
p = size(Xdata,2);
N = size(Xdata,1);

mean_xd = mean(Data,2);

[~,Binit,~,~,explained] = pca(Xdata-mu_mat*ones(1,p));

for i = 1:p
    
    perc = perc + explained(i);
    if (perc >= percvar)
        k = i;
        break;
    end
    
end

R = 1;
beta = 2.3;
sigmap = zeros(sizeim);
Cinitk = Binit(:,1:k)'*(Xdata-mu_mat*ones(1,p));
errpix = Xdata - mu_mat*ones(1,p)- Binit(:,1:k)*Cinitk;

%median absolute value
errpix2 = errpix(:)- median(abs(errpix(:)));
sigmamin = sqrt(3)*1.4826*median(abs(errpix2(:)));

%try 4by4
for i = R+1:N-R-1
    for j = R+1:p-R-1
        [y,x] = meshgrid(i-R:i+R,j-R:j+R);
        indx = sub2ind(im_sz,y(:),x(:));
        errind = errpix(ind,:);
            errpix2 = errind - median(abs(errind(:)));
                sigmap(i,j)=beta*sqrt(3)*1.4826*median(abs(errpix(:)));
    end
end

sigmap = sigmap(:);
sigmap =  max(sigmap,sigmamin*ones(size(sigmap)));
sigma_end = 3*sigmap;

% Initial PCA Values
murpca = median(Xdata')';
Brpca = Binit(:,1:k);
Crpca = Binit(:,1:k)'*(Xdata - mu_mat*ones(1,p));
max_iter = 100;

[Brob,Crob,sigmarob,mu_rob] = rPCA(Xdata,k,max_iter,sigma_end,sigmap,Brpca,Crpca,murpca);

end