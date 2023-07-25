function scale_reduction_factor = compute_scale_reduction_factor(x2)
% chains: A 3D array containing the MCMC chains. The dimensions should be
%         (chain_length x chain_dimension x num_chains).

% drop 1 observation if there is an odd number of draws
if mod(length(x2),2)==1
    x2(1,:) = [];
end
splitat = length(x2)/2;



chains(:,:,1) = x2(1:splitat,:);
chains(:,:,2) = x2(splitat+1:end,:);

% Get the dimensions of the chains
[chain_length, chain_dimension, num_chains] = size(chains);



% Compute the within-chain variances
within_chain_var = zeros(chain_dimension, num_chains);
for chain = 1:num_chains
    for dimension = 1:chain_dimension
        within_chain_var(dimension, chain) = var(chains(:, dimension, chain));
        within_chain_mean(dimension, chain) = mean(chains(:, dimension, chain));
    end
end
all_chain_mean = mean(within_chain_mean,2);
B = ((chain_length)/(num_chains-1))*sum((within_chain_mean-all_chain_mean).^2,2);
W = (1/num_chains)*sum(within_chain_var,2);

% Compute the estimated variance of the pooled chains
pooled_var = ((chain_length - 1) / chain_length) * W ...
    + (1 / chain_length) * B;

% Compute the scale reduction factor
scale_reduction_factor = sqrt(pooled_var ./ W);


end