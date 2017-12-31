function [mu_rob,Brob,Crob,Wopt,evo] = rPCA(Xdata,im_sz,percvar)

%collecting info
p = size(Xdata,2);
N = size(Xdata,1);
perc = 0;

mu_mat = mean(Xdata,2);

[~,Binit,~,~,explained] = pca(Xdata - mu_mat);

for i = 1:p
    
    perc = perc + explained(i);
    if (perc >= percvar)
        k = i;
        break;
    end
    
end


R = 1;
beta = 2.3;
sigmap = zeros(im_sz);
Cinitk = Binit(:,1:k)'*(Xdata - mu_mat*ones(1,p));
errpix = Xdata - mu_mat*ones(1,p)- Binit(:,1:k)*Cinitk;

%median absolute value
errpix2 = errpix(:)- median(abs(errpix(:)));
sigmamin = sqrt(3)*1.4826*median(abs(errpix2(:)));

%try 4by4
for i = R+1:im_sz(1)-R-1
    for j = R+1:im_sz(2)-R-1
        [y,x] = meshgrid(i-R:i+R,j-R:j+R);
        indx = sub2ind(im_sz,y(:),x(:));
        errind = errpix(indx,:);
            errpix2 = errind - median(abs(errind(:)));
                sigmap(i,j)=beta*sqrt(3)*1.4826*median(abs(errpix2(:)));
    end
end

sigmap = sigmap(:);
sigmap =  max(sigmap,sigmamin*ones(size(sigmap)));
sigma_end = 3*sigmap;

% Initial PCA Values
mu_rob = median(Xdata')';
Brob = Binit(:,1:k);
Crob = Binit(:,1:k)'*(Xdata - mu_mat*ones(1,p));
max_iter = 100;

evo = zeros(max_iter,2);
Sigma = sigmap;
iter=1;
mu = 1;
errpixtot = 1;
flag = 1;
iter_param = 2;


while iter < max_iter && flag
    
%store past values
sigma_max = max(Sigma);
sigma_min = min(Sigma);
Brob_init = Brob;

% annealing schedule
Sigma = Sigma*0.92;
Sigma = max(Sigma,sigma_end);

sigmatemp = Sigma.^2 *ones(1,size(Xdata,2));

%updating mu
for i=1:iter_param
    
    errpix = (Xdata - mu_rob - Brob*Crob);
    gradperr = (errpix.*sigmatemp)./((sigmatemp + errpix.^2).^2); %gradient of p(Erpca,sigma) with respect to Erpca where p(x,sigma) = x^2/(x^2 + sigma^2)
    mu_rob = mu_rob + mu*(gradperr*ones(size(Xdata,2),1)./(size(Xdata,2))*1./sigmatemp(:,1));
    
end

errpixtot_init = errpixtot;
errpixtot = sum(sum((errpix.^2)./(sigmatemp + errpix.^2)));  % total error of matrix d by N of p(err,sigma)

%updating Brob
for i=1:iter_param
    
    errpix = (Xdata - mu_rob - Brob*Crob);
    gradpB = (errpix.*sigmatemp)./((sigmatemp + errpix.^2).^2);
    Brob = Brob + mu*(gradpB*Crob')./((1./sigmatemp)*(Crob.*Crob)');
    
end

%updating Crob
for i=1:iter_param
    errpix = (Xdata - mu_rob - Brob*Crob);
    gradpC = (errpix.*sigmatemp)./((sigmatemp + errpix.^2).^2);
    Crob = Crob + mu*(Brob'*gradpC)./((Brob.*Brob)'*(1./sigmatemp));
end

ang_err = subspace(Brob,Brob_init);




evo(iter,:)=[errpixtot ang_err];

%annealing schedule for mu
if (errpixtot > errpixtot_init) && (sigma_max == max(Sigma)) && (sigma_min == min(Sigma))
    mu = mu*0.9;
end

if(ang_err>1e-4 && (iter >= 30))
    flag=0;
end

iter = iter + 1;

fprintf('Iter:%d , Err:%.3f ,  angular_error: %.3f \n',iter,errpixtot, ang_err); 

end

errpix = Xdata - mu_rob *ones(1,p) - Brob*Crob;
sigmatemp = (sigma_end * ones(1,p));
temp = abs(errpix) < sigmatemp/sqrt(3);
Wopt = (sigmatemp./((sigmatemp + errpix.^2).^2)).*temp;


end