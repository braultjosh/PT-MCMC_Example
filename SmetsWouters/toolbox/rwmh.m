function [ind_para, ind_logp, ind_loglh, ind_logpr, ind_acpt] = rwmh(p0, logpost0, loglh0, logprior0, fcn, tune, chain_idx)
        
        warning('off','all');
        
        try
            px = mvnrnd(p0',(tune.c(chain_idx)^2).*tune.cov(:,:,chain_idx))';
            [obj, prior]            = feval(fcn, px, tune.M_, tune.estim_params_, tune.oo_, tune.options_, tune.bayestopt_);
    
            if any(px<tune.bounds.lb) || any(px>tune.bounds.ub) || isnan(obj) == 1  || isnan(prior) == 1 || obj == -Inf || prior == -Inf
                    obj          = -1e20;
                    prior        = -1e20;
                    lx           = -1e20;
                    ind_acpt     = 0;
            end
    
            postx                   = obj*tune.temp(chain_idx) + prior;
            alp                     = exp(postx - logpost0); % this is RW, so q is canceled out
        
            if rand < alp % accept
                ind_para    = px;
                ind_logp    = tune.temp(chain_idx)*obj + prior;
                ind_loglh   = obj;
                ind_logpr   = prior;
                ind_acpt    = 1;
            else % reject
                ind_para    = p0;
                ind_logp    = logpost0;
                ind_loglh   = loglh0;
                ind_logpr   = logprior0;
                ind_acpt    = 0;
            end

        catch
            ind_para    = p0;
            ind_logp    = logpost0;
            ind_loglh   = loglh0;
            ind_logpr   = logprior0;
            ind_acpt    = 0;
            fprintf('problem in rwmh')
        end


end
