function [ind_para, ind_logp, ind_loglh, ind_logpr, ind_acpt] = block_rwmh(p0, logpost0, loglh0, logprior0, fcn, tune, chain_idx)
        
        warning('off','all');
        numBlocks = tune.blocks;
        % Randomly choose a block
        blockIdx = randi([1, numBlocks], tune.nparam, 1);

        ind_para = p0;
        ind_logp = logpost0;
        ind_loglh = loglh0;
        ind_logpr = logprior0;
        ind_acpt  = 0;


        for bb = 1:numBlocks

        px                           = ind_para;
        px_prop                      = ind_para + tune.c(chain_idx)^2*tune.Rchol(:,:,chain_idx)*randn(length(p0),1);
        px([blockIdx==bb])           = px_prop([blockIdx==bb]);
        [obj, prior]                 = feval(fcn, px, tune.M_, tune.estim_params_, tune.oo_, tune.options_, tune.bayestopt_);

        if any(px<tune.bounds.lb) || any(px>tune.bounds.ub) || isnan(obj) == 1  || isnan(prior) == 1 || obj == -Inf
                obj          = -1e+20;
                prior        = -1e+20;
                lx           = -1e+20;   
        end

        postx                   = obj*tune.temp(chain_idx) + prior;
        alp                     = exp(postx - logpost0); % this is RW, so q is canceled out
    
        if rand < alp % accept
            ind_para   = px;
            ind_logp   = tune.temp(chain_idx)*obj + prior;
            ind_loglh  = obj;
            ind_logpr  = prior;
            ind_acpt   = ind_acpt + 1;
        end

        end

        ind_acpt   = ind_acpt/numBlocks;

end
