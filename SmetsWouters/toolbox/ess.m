function [neff] = ess(x2)
% Computes the effective sample size (ESS) for a matrix of parameter draws
% Input: draws (M x N) - a matrix of M parameter draws, each with N elements
% Output: ESS - the effective sample size

[n, param]      = size(x2);
m               = 1;
autocorr_vals = zeros(n,size(x2,2));
for i = 1:size(x2,2)
    autocorr_vals(:,i) = autocorr(x2(:,i), Numlags = n-1);
end
autocorr_vals(1,:) = [];
autocorr_sum = movsum(autocorr_vals,2);

for i = 1:size(x2,2)
    temp_stop = find(autocorr_sum(:,i)<0);
    if mod(temp_stop(1)-2,2)==0
        T = temp_stop(1)-2;
    else
        T = temp_stop(1)-1;
    end
    stop(i) = T;
end

for i = 1:size(x2,2)
    neff(i,1) = (m*n)/(1+2*sum(autocorr_vals(1:stop(i),i)));
end

end
