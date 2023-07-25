close all; 
clear; clc;
warning('off','all');
addpath('./toolbox/');

% User specified options %
tune.niter          = 1000000; %  number of iterations
tune.nchain         = 24; % number of auxiliary distributions
tune.nburn          = 250000; % number of burn-in iterations
tune.c              = repmat(0.025, 1, tune.nchain); % initial scale parameter
tune.trgt           = 0.23; % target for MH acceptance
tune.blocks         = 1;
gen_priorcov        = 1; % use prior covariance matrix for RWMH proposal


% Dynare mod file name
ModelName                    = 'Smets_Wouters_2007_45';


% Please take a look at the examples to see how to specify the estimation
% command in the Dynare mod file
dynare ([ModelName]);
[dataset_, dataset_info, ~, ~, M_, options_, oo_, estim_params_,bayestopt_, bounds] = dynare_estimation_init({}, M_.dname, 0, M_, options_, oo_, estim_params_, bayestopt_);
fcn = @evaluate_likelihood2; % this is not a standard Dynare file, rather I have edited this file to also output the prior density

% start diary
diary results.log;

% add Dynare files to tune structure
tune.nparam         = length(bayestopt_.pshape); % number of parameters
tune.M_             = M_;
tune.estim_params_  = estim_params_;
tune.options_       = options_;
tune.bayestopt_     = bayestopt_;
tune.bounds         = bounds;
tune.oo_            = oo_;


% % % % % Generate covariance matrix for proposal density from prior covariance
if gen_priorcov == 1
    parfor j = 1:1:5000000
            pdraw               = prior_draw(bayestopt_, 0); 
            pdraw               = prior_draw();
            priorc_draws(:,j)   = pdraw;
    end
    priorc_draws = priorc_draws';
end
covmat              = cov(priorc_draws);
priorcov            = diag(diag(covmat));
tune.cov            = repmat(priorcov, [1 1 tune.nchain]); 
tune.Rchol          = repmat(chol(priorcov),[1 1 tune.nchain]); % temperature schedule


% % linear temperature ladder
lambda = 1.5;
temp_ladder(tune.nchain) = 1;
for i = tune.nchain-1:-1:1
    temp_ladder(i)  = ((i)/(tune.nchain+0.25))^(lambda);
end
tune.temp   = temp_ladder; % temperature schedule

tune % store tuning information in the log file
% Preallocation
parasim = zeros(tune.nparam, tune.nchain, tune.niter); % parameter draws
acptsim = zeros(tune.niter,tune.nchain); % average acceptance rate
swapsim = NaN(tune.niter,tune.nchain); % 1 if re-sampled
loglh   = zeros(tune.niter,tune.nchain); %log-likelihood
logpost = zeros(tune.niter,tune.nchain); %log-posterior
logprior = zeros(tune.niter,tune.nchain); % log-prior

% Generate initial parameters using draws from priors
for j = 1:1:tune.nchain
        pdraw = prior_draw(bayestopt_, 0); 
        success = 0;
        while success < 1
            pdraw                   = prior_draw();
            [obj, prior]            = feval(fcn, pdraw', M_, estim_params_, oo_, options_, bayestopt_);
            if sum(pdraw'>bounds.lb) == tune.nparam && sum(pdraw'<bounds.ub) == tune.nparam && obj ~= -Inf && prior ~= -Inf
                    mh_param(:,j)         = pdraw;
                    mh_ll(j)                = obj;
                    mh_prior(j)             = prior;
                    mh_logp(j)              = tune.temp(j)*obj+prior;
                    success = success + 1;
            end
        end
end

fprintf('Initial parameter vectors draw from the priors \n')
mh_param

parasim(:,:,1)          = mh_param;
loglh(1,:)              = mh_ll;
logpost(1,:)            = mh_logp;
logprior(1,:)           = mh_prior;

% generate proposed swaps before starting algorithm
cswap = ceil(rand(tune.niter, 2) * tune.nchain);
cswap(cswap(:, 1) == cswap(:, 2), :) = 0;


% start parallel tempering
pttime      = tic;
totaltime   = 0;
disp_counter = 2; counter = 2; 
ii = 2;
fprintf('Starting Parallel Tempering ... \n')
while ii <= tune.niter

    ind_acpt = NaN(1, tune.nchain);

    % Step 1: Swap proposal between chains
    swap1 = cswap(ii, 1);
    swap2 = cswap(ii, 2);
    
    if swap1 ~= 0 && swap2 ~= 0
        Proposal = mh_ll(swap2) * tune.temp(swap1) + mh_prior(swap2) + mh_ll(swap1) * tune.temp(swap2) + mh_prior(swap1);
        Previous = mh_logp(swap1) + mh_logp(swap2);
    
        if rand < exp(Proposal - Previous)
            [mh_logp(swap1), mh_logp(swap2)] = deal(mh_ll(swap2) * tune.temp(swap1) + mh_prior(swap2), mh_ll(swap1) * tune.temp(swap2) + mh_prior(swap1));
            [mh_param(:, [swap1, swap2]), mh_ll(1, [swap1, swap2]), mh_prior(1, [swap1, swap2])] = deal(mh_param(:, [swap2, swap1]), mh_ll(1, [swap2, swap1]), mh_prior(1, [swap2, swap1]));
            if abs(swap1 - swap2) == 1
            swapsim(ii, [swap1, swap2]) = 1;
            end
          else
            if abs(swap1 - swap2) == 1
                swapsim(ii, [swap1, swap2]) = 0;
            end
        end

    end

    % Step 2: Mutate via RWMH
    parfor chain_idx = 1:tune.nchain    
        [mh_param(:,chain_idx), mh_logp(chain_idx), mh_ll(chain_idx), mh_prior(chain_idx), ind_acpt(chain_idx)] = ...
                rwmh(mh_param(:, chain_idx), mh_logp(chain_idx), mh_ll(chain_idx), mh_prior(chain_idx), fcn, tune, chain_idx);

        acptsim(ii, chain_idx)      = ind_acpt(chain_idx);
        parasim(:, chain_idx, ii)   = mh_param(:, chain_idx);
        logpost(ii, chain_idx)      = mh_logp(chain_idx);
        loglh(ii, chain_idx)        = mh_ll(chain_idx);
        logprior(ii, chain_idx)     = mh_prior(chain_idx);
    end 

    % Update scaling parameter if within tuning period
    if counter >= 1000 && ii < tune.nburn
        acpt_mean = mean(acptsim(ii-999:ii,:));
        exp_term = exp(16 * (acpt_mean - tune.trgt));
        tune.c = tune.c .*(0.95 + ((0.10 * exp_term) ./ (1 + exp_term)));
        counter = 0;
    end
    

    % Update covariance matrix for proposal density if equal to iteration #
    if ii == 200000
        for i = 1:tune.nchain
            parasim_reshaped = squeeze(parasim(:, i, 1:ii));  % Reshape parasim for covariance calculation
            sample_cov = cov(parasim_reshaped');
            local_corr = sample_cov ./ sqrt(diag(sample_cov)) ./ sqrt(diag(sample_cov)).';  % Calculate local correlations
            sample_cov(abs(local_corr) < 0.1) = 0;  % Zero out elements based on correlation threshold
            sample_cov = (sample_cov + sample_cov')/2; % ensure symmetry
            try
            chol(sample_cov);
            tune.cov(:,:,i) = sample_cov;
            catch
            tune.cov(:,:,i) = nearestSPD(sample_cov);
            end
        end
    end


    if disp_counter>=10000
        totaltime = totaltime + toc(pttime);
        avgtime   = totaltime/ii;
        remtime   = avgtime*(tune.niter-ii);

        fprintf('-----------------------------------------------\n')
        fprintf(' Iteration = %10.0f \n', ii)
        fprintf(' Results over last 10000 iterations for target post \n')
        fprintf('-----------------------------------------------\n')
        fprintf('  Time elapsed   = %.3f\n', totaltime)
        fprintf('  Time average   = %.3f\n', avgtime)
        fprintf('  Time remained  = %.3f\n', remtime)
        for c = tune.nchain:tune.nchain % useful
        %for c = 1:tune.nchain % use this to see intermediate
        %results for all chains
            fprintf('-----------------------------------------------\n')
            fprintf('For chain %5.2f \n', c)
            fprintf('-----------------------------------------------\n')
            fprintf('  Average log posterior   = %5.2f\n',  mean(logpost(ii-9999:ii,c)))
            fprintf('  Average mh accept   = %.3f\n',  mean(acptsim(ii-9999:ii,c),'omitnan'))
            fprintf('  Average swap accept  = %.3f\n',  mean(swapsim(ii-9999:ii,c),'omitnan'))
            fprintf('-----------------------------------------------\n')
            fprintf('%-20s   %s  %s \n', 'para', 'mean', 'std')
            fprintf('%-20s   %s  %s \n','------'   , '----',    '----')
           for n = 1:tune.nparam
                    fprintf('%-20s   %.3f  %.3f  \n', bayestopt_.name{n}, mean(squeeze(parasim(n,c,ii-9999:ii))), std(squeeze(parasim(n,c,ii-9999:ii))) )
           end
       end
       disp_counter = 0;
       pttime = tic;
    end

    disp_counter       = disp_counter + 1;
    counter            = counter + 1;
    ii = ii +1;
end

%=========================================================================
%                  Some Posterior Analysis
%=========================================================================

tune
x2 = squeeze(parasim(:,tune.nchain,(tune.nburn+1):tune.niter))';
logpo2 = logpost((tune.nburn+1):tune.niter,tune.nchain);


%=========================================================================
%                  Compute posterior moments
%=========================================================================
for jj = 1:tune.nparam
[post_mean(jj), post_median(jj), post_var(jj), hpd_interval(jj,:)] = posterior_moments(x2(:,jj),1,0.90);
end

%=========================================================================
%                  Compute marginal density
%=========================================================================
[T,npar] = size(x2);

[tmp,idx] = max(logpo2);
posterior_mode = x2(idx,:);
posterior_kernel_at_the_mode = tmp;
lpost_mode = posterior_kernel_at_the_mode;

mu = mean(x2);
sigma = cov(x2);
logdetsigma = log(det(sigma));
invsigma = inv(sigma);
marginal = zeros(9,2);
increase = 1;

linee = 0;
for p = 0.1:0.1:0.9
    critval = chi2inv(p,npar);
    tmp = 0;
    for i = 1:length(x2)
        deviation  = ((x2(i,:)-mu)*invsigma*(x2(i,:)-mu)')/increase;
      if deviation <= critval
        lftheta = -log(p)-(npar*log(2*pi)+(npar*log(increase)+logdetsigma)+deviation)/2;
        tmp = tmp + exp(lftheta - logpo2(i) + lpost_mode);
      end
    end
    linee = linee + 1;
    marginal(linee,:) = [p, lpost_mode-log(tmp/((tune.niter-tune.nburn)*1))];
end
vartype = bayestopt_.name;
sum_vec = round([post_mean', hpd_interval(:,1), hpd_interval(:,2)],3);

disp('                                                                          '); 
disp('                                                                          ');
disp('                                                                          ');
disp('                                                                          ');
disp('                                                                          ');
disp('                                                                          ');
disp('                                                                          ');
disp('                                                                          ');
disp('========================================================================='); 
disp(['Posterior results']);
disp('========================================================================='); 
fprintf('  Time elapsed   = %5.2f\n', totaltime)
disp('=========================================================================');
fprintf('  Average log posterior (hot to cold)   = %5.2f\n',  mean(logpost(tune.nburn:tune.niter,:)))
fprintf('  Average mh accept  (hot to cold)  = %5.2f\n',  mean(acptsim(tune.nburn:tune.niter,:),'omitnan'))
fprintf('  Average swap accept (hot to cold)  = %5.2f\n',  mean(swapsim(tune.nburn:tune.niter,:),'omitnan'))
disp('=========================================================================');
disp('=========================================================================');
disp(' Variable Name                       Mean         5%        95%         ');
disp('=========================================================================');
for hh=1:length(vartype);
    fprintf('%-30s %10.4f %10.4f %10.4f\n',vartype{hh},sum_vec(hh,1),...
        sum_vec(hh,2),sum_vec(hh,3));    
end
disp('========================================================================='); 
disp(['Marginal data density    : ', num2str(mean(marginal(:,2)))]);
disp('========================================================================='); 
% disp('========================================================================='); 
% fprintf(' Probability of determinacy = %5.4f\n', sum(x2(:,20)>1)/length(x2));
% disp('========================================================================='); 

%=========================================================================
%                  Compute effective sample size (ESS)
%=========================================================================
disp('=========================================================================');
disp(' Variable Name                       ESS        ');
disp('=========================================================================');
EffSamp = ess(x2);
for hh=1:length(vartype);
    fprintf('%-30s %10.4f\n',vartype{hh},round(EffSamp(hh,1)));    
end

%=========================================================================
%                  Convergence diagnostic
%=========================================================================
disp('=========================================================================');
disp(' Variable Name                       Scale Reduction Factor        ');
disp('=========================================================================');
R = compute_scale_reduction_factor(x2);
for hh=1:length(vartype);
    fprintf('%-30s %10.4f\n',vartype{hh},R(hh,1));    
end

%=========================================================================
%                  FIGURE 2: POSTERIOR MARGINAL DENSITIES 
%=========================================================================
pnames = strvcat(bayestopt_.name);
figure('Position',[20,20,900,600],'Name',...
    'Posterior Marginal Densities','Color','w')
for i=1:(npar)
    xmin = min(x2(:,i));
    xmax = max(x2(:,i));
    grid = linspace(xmin,xmax,100);
    u    = (1+0.4)*max(ksdensity(x2(:,i)));
    subplot(ceil((npar)/3),3,i), plot(grid,ksdensity(x2(:,i)),'LineStyle','-','Color','b',...
            'LineWidth',2.5), hold on
    plot([mean(x2(:,i)) mean(x2(:,i))], [0 u],'LineStyle',':',...
        'Color','black','LineWidth',2.5 ), hold off
axis([xmin xmax 0 u]);
title(pnames(i,:),'FontSize',12,'FontWeight','bold');    
end


diary off;
