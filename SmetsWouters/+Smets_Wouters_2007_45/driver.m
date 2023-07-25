%
% Status : main Dynare file
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

if isoctave || matlab_ver_less_than('8.6')
    clear all
else
    clearvars -global
    clear_persistent_variables(fileparts(which('dynare')), false)
end
tic0 = tic;
% Define global variables.
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info ys0_ ex0_
options_ = [];
M_.fname = 'Smets_Wouters_2007_45';
M_.dynare_version = '5.2';
oo_.dynare_version = '5.2';
options_.dynare_version = '5.2';
%
% Some global variables initialization
%
global_initialization;
M_.exo_names = cell(7,1);
M_.exo_names_tex = cell(7,1);
M_.exo_names_long = cell(7,1);
M_.exo_names(1) = {'ea'};
M_.exo_names_tex(1) = {'{\eta^a}'};
M_.exo_names_long(1) = {'productivity shock'};
M_.exo_names(2) = {'eb'};
M_.exo_names_tex(2) = {'{\eta^b}'};
M_.exo_names_long(2) = {'risk premium shock'};
M_.exo_names(3) = {'eg'};
M_.exo_names_tex(3) = {'{\eta^g}'};
M_.exo_names_long(3) = {'Spending shock'};
M_.exo_names(4) = {'eqs'};
M_.exo_names_tex(4) = {'{\eta^i}'};
M_.exo_names_long(4) = {'Investment-specific technology shock'};
M_.exo_names(5) = {'em'};
M_.exo_names_tex(5) = {'{\eta^m}'};
M_.exo_names_long(5) = {'Monetary policy shock'};
M_.exo_names(6) = {'epinf'};
M_.exo_names_tex(6) = {'{\eta^{p}}'};
M_.exo_names_long(6) = {'Price markup shock'};
M_.exo_names(7) = {'ew'};
M_.exo_names_tex(7) = {'{\eta^{w}}'};
M_.exo_names_long(7) = {'Wage markup shock'};
M_.endo_names = cell(40,1);
M_.endo_names_tex = cell(40,1);
M_.endo_names_long = cell(40,1);
M_.endo_names(1) = {'labobs'};
M_.endo_names_tex(1) = {'{lHOURS}'};
M_.endo_names_long(1) = {'log hours worked'};
M_.endo_names(2) = {'robs'};
M_.endo_names_tex(2) = {'{FEDFUNDS}'};
M_.endo_names_long(2) = {'Federal funds rate'};
M_.endo_names(3) = {'pinfobs'};
M_.endo_names_tex(3) = {'{dlP}'};
M_.endo_names_long(3) = {'Inflation'};
M_.endo_names(4) = {'dy'};
M_.endo_names_tex(4) = {'{dlGDP}'};
M_.endo_names_long(4) = {'Output growth rate'};
M_.endo_names(5) = {'dc'};
M_.endo_names_tex(5) = {'{dlCONS}'};
M_.endo_names_long(5) = {'Consumption growth rate'};
M_.endo_names(6) = {'dinve'};
M_.endo_names_tex(6) = {'{dlINV}'};
M_.endo_names_long(6) = {'Investment growth rate'};
M_.endo_names(7) = {'dw'};
M_.endo_names_tex(7) = {'{dlWAG}'};
M_.endo_names_long(7) = {'Wage growth rate'};
M_.endo_names(8) = {'ewma'};
M_.endo_names_tex(8) = {'{\eta^{w,aux}}'};
M_.endo_names_long(8) = {'Auxiliary wage markup moving average variable'};
M_.endo_names(9) = {'epinfma'};
M_.endo_names_tex(9) = {'{\eta^{p,aux}}'};
M_.endo_names_long(9) = {'Auxiliary price markup moving average variable'};
M_.endo_names(10) = {'zcapf'};
M_.endo_names_tex(10) = {'{z^{flex}}'};
M_.endo_names_long(10) = {'Capital utilization rate flex price economy'};
M_.endo_names(11) = {'rkf'};
M_.endo_names_tex(11) = {'{r^{k,flex}}'};
M_.endo_names_long(11) = {'rental rate of capital flex price economy'};
M_.endo_names(12) = {'kf'};
M_.endo_names_tex(12) = {'{k^{s,flex}}'};
M_.endo_names_long(12) = {'Capital services flex price economy'};
M_.endo_names(13) = {'pkf'};
M_.endo_names_tex(13) = {'{q^{flex}}'};
M_.endo_names_long(13) = {'real value of existing capital stock flex price economy'};
M_.endo_names(14) = {'cf'};
M_.endo_names_tex(14) = {'{c^{flex}}'};
M_.endo_names_long(14) = {'Consumption flex price economy'};
M_.endo_names(15) = {'invef'};
M_.endo_names_tex(15) = {'{i^{flex}}'};
M_.endo_names_long(15) = {'Investment flex price economy'};
M_.endo_names(16) = {'yf'};
M_.endo_names_tex(16) = {'{y^{flex}}'};
M_.endo_names_long(16) = {'Output flex price economy'};
M_.endo_names(17) = {'labf'};
M_.endo_names_tex(17) = {'{l^{flex}}'};
M_.endo_names_long(17) = {'hours worked flex price economy'};
M_.endo_names(18) = {'wf'};
M_.endo_names_tex(18) = {'{w^{flex}}'};
M_.endo_names_long(18) = {'real wage flex price economy'};
M_.endo_names(19) = {'rrf'};
M_.endo_names_tex(19) = {'{r^{flex}}'};
M_.endo_names_long(19) = {'real interest rate flex price economy'};
M_.endo_names(20) = {'mc'};
M_.endo_names_tex(20) = {'{\mu_p}'};
M_.endo_names_long(20) = {'gross price markup'};
M_.endo_names(21) = {'zcap'};
M_.endo_names_tex(21) = {'{z}'};
M_.endo_names_long(21) = {'Capital utilization rate'};
M_.endo_names(22) = {'rk'};
M_.endo_names_tex(22) = {'{r^{k}}'};
M_.endo_names_long(22) = {'rental rate of capital'};
M_.endo_names(23) = {'k'};
M_.endo_names_tex(23) = {'{k^{s}}'};
M_.endo_names_long(23) = {'Capital services'};
M_.endo_names(24) = {'pk'};
M_.endo_names_tex(24) = {'{q}'};
M_.endo_names_long(24) = {'real value of existing capital stock'};
M_.endo_names(25) = {'c'};
M_.endo_names_tex(25) = {'{c}'};
M_.endo_names_long(25) = {'Consumption'};
M_.endo_names(26) = {'inve'};
M_.endo_names_tex(26) = {'{i}'};
M_.endo_names_long(26) = {'Investment'};
M_.endo_names(27) = {'y'};
M_.endo_names_tex(27) = {'{y}'};
M_.endo_names_long(27) = {'Output'};
M_.endo_names(28) = {'lab'};
M_.endo_names_tex(28) = {'{l}'};
M_.endo_names_long(28) = {'hours worked'};
M_.endo_names(29) = {'pinf'};
M_.endo_names_tex(29) = {'{\pi}'};
M_.endo_names_long(29) = {'Inflation'};
M_.endo_names(30) = {'w'};
M_.endo_names_tex(30) = {'{w}'};
M_.endo_names_long(30) = {'real wage'};
M_.endo_names(31) = {'r'};
M_.endo_names_tex(31) = {'{r}'};
M_.endo_names_long(31) = {'nominal interest rate'};
M_.endo_names(32) = {'a'};
M_.endo_names_tex(32) = {'{\varepsilon_a}'};
M_.endo_names_long(32) = {'productivity process'};
M_.endo_names(33) = {'b'};
M_.endo_names_tex(33) = {'{c_2*\varepsilon_t^b}'};
M_.endo_names_long(33) = {'Scaled risk premium shock'};
M_.endo_names(34) = {'g'};
M_.endo_names_tex(34) = {'{\varepsilon^g}'};
M_.endo_names_long(34) = {'Exogenous spending'};
M_.endo_names(35) = {'qs'};
M_.endo_names_tex(35) = {'{\varepsilon^i}'};
M_.endo_names_long(35) = {'Investment-specific technology'};
M_.endo_names(36) = {'ms'};
M_.endo_names_tex(36) = {'{\varepsilon^r}'};
M_.endo_names_long(36) = {'Monetary policy shock process'};
M_.endo_names(37) = {'spinf'};
M_.endo_names_tex(37) = {'{\varepsilon^p}'};
M_.endo_names_long(37) = {'Price markup shock process'};
M_.endo_names(38) = {'sw'};
M_.endo_names_tex(38) = {'{\varepsilon^w}'};
M_.endo_names_long(38) = {'Wage markup shock process'};
M_.endo_names(39) = {'kpf'};
M_.endo_names_tex(39) = {'{k^{flex}}'};
M_.endo_names_long(39) = {'Capital stock flex price economy'};
M_.endo_names(40) = {'kp'};
M_.endo_names_tex(40) = {'{k}'};
M_.endo_names_long(40) = {'Capital stock'};
M_.endo_partitions = struct();
M_.param_names = cell(39,1);
M_.param_names_tex = cell(39,1);
M_.param_names_long = cell(39,1);
M_.param_names(1) = {'curvw'};
M_.param_names_tex(1) = {'{\varepsilon_w}'};
M_.param_names_long(1) = {'Curvature Kimball aggregator wages'};
M_.param_names(2) = {'cgy'};
M_.param_names_tex(2) = {'{\rho_{ga}}'};
M_.param_names_long(2) = {'Feedback technology on exogenous spending'};
M_.param_names(3) = {'curvp'};
M_.param_names_tex(3) = {'{\varepsilon_p}'};
M_.param_names_long(3) = {'Curvature Kimball aggregator prices'};
M_.param_names(4) = {'constelab'};
M_.param_names_tex(4) = {'{\bar l}'};
M_.param_names_long(4) = {'steady state hours'};
M_.param_names(5) = {'constepinf'};
M_.param_names_tex(5) = {'{\bar \pi}'};
M_.param_names_long(5) = {'steady state inflation rate'};
M_.param_names(6) = {'constebeta'};
M_.param_names_tex(6) = {'{100(\beta^{-1}-1)}'};
M_.param_names_long(6) = {'time preference rate in percent'};
M_.param_names(7) = {'cmaw'};
M_.param_names_tex(7) = {'{\mu_w}'};
M_.param_names_long(7) = {'coefficient on MA term wage markup'};
M_.param_names(8) = {'cmap'};
M_.param_names_tex(8) = {'{\mu_p}'};
M_.param_names_long(8) = {'coefficient on MA term price markup'};
M_.param_names(9) = {'calfa'};
M_.param_names_tex(9) = {'{\alpha}'};
M_.param_names_long(9) = {'capital share'};
M_.param_names(10) = {'czcap'};
M_.param_names_tex(10) = {'{\psi}'};
M_.param_names_long(10) = {'capacity utilization cost'};
M_.param_names(11) = {'csadjcost'};
M_.param_names_tex(11) = {'{\varphi}'};
M_.param_names_long(11) = {'investment adjustment cost'};
M_.param_names(12) = {'ctou'};
M_.param_names_tex(12) = {'{\delta}'};
M_.param_names_long(12) = {'depreciation rate'};
M_.param_names(13) = {'csigma'};
M_.param_names_tex(13) = {'{\sigma_c}'};
M_.param_names_long(13) = {'risk aversion'};
M_.param_names(14) = {'chabb'};
M_.param_names_tex(14) = {'{\lambda}'};
M_.param_names_long(14) = {'external habit degree'};
M_.param_names(15) = {'ccs'};
M_.param_names_tex(15) = {'{d_4}'};
M_.param_names_long(15) = {'Unused parameter'};
M_.param_names(16) = {'cinvs'};
M_.param_names_tex(16) = {'{d_3}'};
M_.param_names_long(16) = {'Unused parameter'};
M_.param_names(17) = {'cfc'};
M_.param_names_tex(17) = {'{\phi_p}'};
M_.param_names_long(17) = {'fixed cost share'};
M_.param_names(18) = {'cindw'};
M_.param_names_tex(18) = {'{\iota_w}'};
M_.param_names_long(18) = {'Indexation to past wages'};
M_.param_names(19) = {'cprobw'};
M_.param_names_tex(19) = {'{\xi_w}'};
M_.param_names_long(19) = {'Calvo parameter wages'};
M_.param_names(20) = {'cindp'};
M_.param_names_tex(20) = {'{\iota_p}'};
M_.param_names_long(20) = {'Indexation to past prices'};
M_.param_names(21) = {'cprobp'};
M_.param_names_tex(21) = {'{\xi_p}'};
M_.param_names_long(21) = {'Calvo parameter prices'};
M_.param_names(22) = {'csigl'};
M_.param_names_tex(22) = {'{\sigma_l}'};
M_.param_names_long(22) = {'Frisch elasticity'};
M_.param_names(23) = {'clandaw'};
M_.param_names_tex(23) = {'{\phi_w}'};
M_.param_names_long(23) = {'Gross markup wages'};
M_.param_names(24) = {'crdpi'};
M_.param_names_tex(24) = {'{r_{\Delta \pi}}'};
M_.param_names_long(24) = {'Unused parameter'};
M_.param_names(25) = {'crpi'};
M_.param_names_tex(25) = {'{r_{\pi}}'};
M_.param_names_long(25) = {'Taylor rule inflation feedback'};
M_.param_names(26) = {'crdy'};
M_.param_names_tex(26) = {'{r_{\Delta y}}'};
M_.param_names_long(26) = {'Taylor rule output growth feedback'};
M_.param_names(27) = {'cry'};
M_.param_names_tex(27) = {'{r_{y}}'};
M_.param_names_long(27) = {'Taylor rule output level feedback'};
M_.param_names(28) = {'crr'};
M_.param_names_tex(28) = {'{\rho}'};
M_.param_names_long(28) = {'interest rate persistence'};
M_.param_names(29) = {'crhoa'};
M_.param_names_tex(29) = {'{\rho_a}'};
M_.param_names_long(29) = {'persistence productivity shock'};
M_.param_names(30) = {'crhoas'};
M_.param_names_tex(30) = {'{d_2}'};
M_.param_names_long(30) = {'Unused parameter'};
M_.param_names(31) = {'crhob'};
M_.param_names_tex(31) = {'{\rho_b}'};
M_.param_names_long(31) = {'persistence risk premium shock'};
M_.param_names(32) = {'crhog'};
M_.param_names_tex(32) = {'{\rho_g}'};
M_.param_names_long(32) = {'persistence spending shock'};
M_.param_names(33) = {'crhols'};
M_.param_names_tex(33) = {'{d_1}'};
M_.param_names_long(33) = {'Unused parameter'};
M_.param_names(34) = {'crhoqs'};
M_.param_names_tex(34) = {'{\rho_i}'};
M_.param_names_long(34) = {'persistence risk premium shock'};
M_.param_names(35) = {'crhoms'};
M_.param_names_tex(35) = {'{\rho_r}'};
M_.param_names_long(35) = {'persistence monetary policy shock'};
M_.param_names(36) = {'crhopinf'};
M_.param_names_tex(36) = {'{\rho_p}'};
M_.param_names_long(36) = {'persistence price markup shock'};
M_.param_names(37) = {'crhow'};
M_.param_names_tex(37) = {'{\rho_w}'};
M_.param_names_long(37) = {'persistence wage markup shock'};
M_.param_names(38) = {'ctrend'};
M_.param_names_tex(38) = {'{\bar \gamma}'};
M_.param_names_long(38) = {'net growth rate in percent'};
M_.param_names(39) = {'cg'};
M_.param_names_tex(39) = {'{\frac{\bar g}{\bar y}}'};
M_.param_names_long(39) = {'steady state exogenous spending share'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 7;
M_.endo_nbr = 40;
M_.param_nbr = 39;
M_.orig_endo_nbr = 40;
M_.aux_vars = [];
options_.varobs = cell(7, 1);
options_.varobs(1)  = {'dy'};
options_.varobs(2)  = {'dc'};
options_.varobs(3)  = {'dinve'};
options_.varobs(4)  = {'labobs'};
options_.varobs(5)  = {'pinfobs'};
options_.varobs(6)  = {'dw'};
options_.varobs(7)  = {'robs'};
options_.varobs_id = [ 4 5 6 1 3 7 2  ];
M_ = setup_solvers(M_);
M_.Sigma_e = zeros(7, 7);
M_.Correlation_matrix = eye(7, 7);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = true;
M_.det_shocks = [];
M_.surprise_shocks = [];
M_.heteroskedastic_shocks.Qvalue_orig = [];
M_.heteroskedastic_shocks.Qscale_orig = [];
options_.linear = true;
options_.block = false;
options_.bytecode = false;
options_.use_dll = false;
M_.nonzero_hessian_eqs = [];
M_.hessian_eq_zero = isempty(M_.nonzero_hessian_eqs);
M_.orig_eq_nbr = 40;
M_.eq_nbr = 40;
M_.ramsey_eq_nbr = 0;
M_.set_auxiliary_variables = exist(['./+' M_.fname '/set_auxiliary_variables.m'], 'file') == 2;
M_.epilogue_names = {};
M_.epilogue_var_list_ = {};
M_.orig_maximum_endo_lag = 1;
M_.orig_maximum_endo_lead = 1;
M_.orig_maximum_exo_lag = 0;
M_.orig_maximum_exo_lead = 0;
M_.orig_maximum_exo_det_lag = 0;
M_.orig_maximum_exo_det_lead = 0;
M_.orig_maximum_lag = 1;
M_.orig_maximum_lead = 1;
M_.orig_maximum_lag_with_diffs_expanded = 1;
M_.lead_lag_incidence = [
 0 21 0;
 0 22 0;
 0 23 0;
 0 24 0;
 0 25 0;
 0 26 0;
 0 27 0;
 1 28 0;
 2 29 0;
 0 30 0;
 0 31 61;
 0 32 0;
 0 33 62;
 3 34 63;
 4 35 64;
 5 36 0;
 0 37 65;
 0 38 0;
 0 39 0;
 0 40 0;
 0 41 0;
 0 42 66;
 0 43 0;
 0 44 67;
 6 45 68;
 7 46 69;
 8 47 0;
 0 48 70;
 9 49 71;
 10 50 72;
 11 51 0;
 12 52 0;
 13 53 0;
 14 54 0;
 15 55 0;
 16 56 0;
 17 57 0;
 18 58 0;
 19 59 0;
 20 60 0;]';
M_.nstatic = 14;
M_.nfwrd   = 6;
M_.npred   = 14;
M_.nboth   = 6;
M_.nsfwrd   = 12;
M_.nspred   = 20;
M_.ndynamic   = 26;
M_.dynamic_tmp_nbr = [18; 0; 0; 0; ];
M_.model_local_variables_dynamic_tt_idxs = {
};
M_.equations_tags = {
  1 , 'name' , 'FOC labor with mpl expressed as function of rk and w, flex price economy' ;
  2 , 'name' , 'FOC capacity utilization, flex price economy' ;
  3 , 'name' , 'Firm FOC capital, flex price economy' ;
  4 , 'name' , 'Definition capital services, flex price economy' ;
  5 , 'name' , 'Investment Euler Equation, flex price economy' ;
  6 , 'name' , 'Arbitrage equation value of capital, flex price economy' ;
  7 , 'name' , 'Consumption Euler Equation, flex price economy' ;
  8 , 'name' , 'Aggregate Resource Constraint, flex price economy' ;
  9 , 'name' , 'Aggregate Production Function, flex price economy' ;
  10 , 'name' , 'Wage equation, flex price economy' ;
  11 , 'name' , 'Law of motion for capital, flex price economy (see header notes)' ;
  12 , 'name' , 'FOC labor with mpl expressed as function of rk and w, SW Equation (9)' ;
  13 , 'name' , 'FOC capacity utilization, SW Equation (7)' ;
  14 , 'name' , 'Firm FOC capital, SW Equation (11)' ;
  15 , 'name' , 'Definition capital services, SW Equation (6)' ;
  16 , 'name' , 'Investment Euler Equation, SW Equation (3)' ;
  17 , 'name' , 'Arbitrage equation value of capital, SW Equation (4)' ;
  18 , 'name' , 'Consumption Euler Equation, SW Equation (2)' ;
  19 , 'name' , 'Aggregate Resource Constraint, SW Equation (1)' ;
  20 , 'name' , 'Aggregate Production Function, SW Equation (5)' ;
  21 , 'name' , 'New Keynesian Phillips Curve, SW Equation (10)' ;
  22 , 'name' , 'Wage Phillips Curve, SW Equation (13), with (12) plugged for mu_w' ;
  23 , 'name' , 'Taylor rule, SW Equation (14)' ;
  24 , 'name' , 'Law of motion for productivity' ;
  25 , 'name' , 'Law of motion for risk premium' ;
  26 , 'name' , 'Law of motion for spending process' ;
  27 , 'name' , 'Law of motion for investment specific technology shock process' ;
  28 , 'name' , 'Law of motion for monetary policy shock process' ;
  29 , 'name' , 'Law of motion for price markup shock process' ;
  30 , 'name' , 'epinfma' ;
  31 , 'name' , 'Law of motion for wage markup shock process' ;
  32 , 'name' , 'ewma' ;
  33 , 'name' , 'Law of motion for capital, SW Equation (8) (see header notes)' ;
  34 , 'name' , 'Observation equation output' ;
  35 , 'name' , 'Observation equation consumption' ;
  36 , 'name' , 'Observation equation investment' ;
  37 , 'name' , 'Observation equation real wage' ;
  38 , 'name' , 'Observation equation inflation' ;
  39 , 'name' , 'Observation equation interest rate' ;
  40 , 'name' , 'Observation equation hours worked' ;
};
M_.mapping.labobs.eqidx = [40 ];
M_.mapping.robs.eqidx = [39 ];
M_.mapping.pinfobs.eqidx = [38 ];
M_.mapping.dy.eqidx = [34 ];
M_.mapping.dc.eqidx = [35 ];
M_.mapping.dinve.eqidx = [36 ];
M_.mapping.dw.eqidx = [37 ];
M_.mapping.ewma.eqidx = [31 32 ];
M_.mapping.epinfma.eqidx = [29 30 ];
M_.mapping.zcapf.eqidx = [2 4 8 ];
M_.mapping.rkf.eqidx = [1 2 3 6 ];
M_.mapping.kf.eqidx = [3 4 9 ];
M_.mapping.pkf.eqidx = [5 6 ];
M_.mapping.cf.eqidx = [7 8 10 ];
M_.mapping.invef.eqidx = [5 8 11 ];
M_.mapping.yf.eqidx = [8 9 23 ];
M_.mapping.labf.eqidx = [3 7 9 10 ];
M_.mapping.wf.eqidx = [1 3 10 ];
M_.mapping.rrf.eqidx = [6 7 ];
M_.mapping.mc.eqidx = [12 21 ];
M_.mapping.zcap.eqidx = [13 15 19 ];
M_.mapping.rk.eqidx = [12 13 14 17 ];
M_.mapping.k.eqidx = [14 15 20 ];
M_.mapping.pk.eqidx = [16 17 ];
M_.mapping.c.eqidx = [18 19 22 35 ];
M_.mapping.inve.eqidx = [16 19 33 36 ];
M_.mapping.y.eqidx = [19 20 23 34 ];
M_.mapping.lab.eqidx = [14 18 20 22 40 ];
M_.mapping.pinf.eqidx = [17 18 21 22 23 38 ];
M_.mapping.w.eqidx = [12 14 22 37 ];
M_.mapping.r.eqidx = [17 18 23 39 ];
M_.mapping.a.eqidx = [1 9 12 20 24 ];
M_.mapping.b.eqidx = [6 7 17 18 25 ];
M_.mapping.g.eqidx = [8 19 26 ];
M_.mapping.qs.eqidx = [5 11 16 27 33 ];
M_.mapping.ms.eqidx = [23 28 ];
M_.mapping.spinf.eqidx = [21 29 ];
M_.mapping.sw.eqidx = [22 31 ];
M_.mapping.kpf.eqidx = [4 11 ];
M_.mapping.kp.eqidx = [15 33 ];
M_.mapping.ea.eqidx = [24 26 ];
M_.mapping.eb.eqidx = [25 ];
M_.mapping.eg.eqidx = [26 ];
M_.mapping.eqs.eqidx = [27 ];
M_.mapping.em.eqidx = [28 ];
M_.mapping.epinf.eqidx = [30 ];
M_.mapping.ew.eqidx = [32 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.state_var = [8 9 14 15 16 25 26 27 29 30 31 32 33 34 35 36 37 38 39 40 ];
M_.exo_names_orig_ord = [1:7];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(40, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(7, 1);
M_.params = NaN(39, 1);
M_.endo_trends = struct('deflator', cell(40, 1), 'log_deflator', cell(40, 1), 'growth_factor', cell(40, 1), 'log_growth_factor', cell(40, 1));
M_.NNZDerivatives = [160; 0; -1; ];
M_.static_tmp_nbr = [16; 2; 0; 0; ];
M_.model_local_variables_static_tt_idxs = {
};
M_.params(12) = .025;
ctou = M_.params(12);
M_.params(23) = 1.5;
clandaw = M_.params(23);
M_.params(39) = 0.18;
cg = M_.params(39);
M_.params(3) = 10;
curvp = M_.params(3);
M_.params(1) = 10;
curvw = M_.params(1);
M_.params(9) = .24;
calfa = M_.params(9);
cbeta=.9995;
M_.params(13) = 1.5;
csigma = M_.params(13);
M_.params(17) = 1.5;
cfc = M_.params(17);
M_.params(2) = 0.51;
cgy = M_.params(2);
M_.params(11) = 6.0144;
csadjcost = M_.params(11);
M_.params(14) = 0.6361;
chabb = M_.params(14);
M_.params(19) = 0.8087;
cprobw = M_.params(19);
M_.params(22) = 1.9423;
csigl = M_.params(22);
M_.params(21) = 0.6;
cprobp = M_.params(21);
M_.params(18) = 0.3243;
cindw = M_.params(18);
M_.params(20) = 0.47;
cindp = M_.params(20);
M_.params(10) = 0.2696;
czcap = M_.params(10);
M_.params(25) = 1.488;
crpi = M_.params(25);
M_.params(28) = 0.8762;
crr = M_.params(28);
M_.params(27) = 0.0593;
cry = M_.params(27);
M_.params(26) = 0.2347;
crdy = M_.params(26);
M_.params(29) = 0.9977;
crhoa = M_.params(29);
M_.params(31) = 0.5799;
crhob = M_.params(31);
M_.params(32) = 0.9957;
crhog = M_.params(32);
M_.params(33) = 0.9928;
crhols = M_.params(33);
M_.params(34) = 0.7165;
crhoqs = M_.params(34);
M_.params(30) = 1;
crhoas = M_.params(30);
M_.params(35) = 0;
crhoms = M_.params(35);
M_.params(36) = 0;
crhopinf = M_.params(36);
M_.params(37) = 0;
crhow = M_.params(37);
M_.params(8) = 0.2;
cmap = M_.params(8);
M_.params(7) = 0.2;
cmaw = M_.params(7);
M_.params(4) = 0;
constelab = M_.params(4);
M_.params(5) = 0.7;
constepinf = M_.params(5);
M_.params(6) = 0.4;
constebeta = M_.params(6);
M_.params(38) = 0.3982;
ctrend = M_.params(38);
%
% SHOCKS instructions
%
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = (0.4618)^2;
M_.Sigma_e(2, 2) = (1.8513)^2;
M_.Sigma_e(3, 3) = (0.6090)^2;
M_.Sigma_e(4, 4) = (0.6017)^2;
M_.Sigma_e(5, 5) = (0.2397)^2;
M_.Sigma_e(6, 6) = (0.1455)^2;
M_.Sigma_e(7, 7) = (0.2089)^2;
estim_params_.var_exo = zeros(0, 10);
estim_params_.var_endo = zeros(0, 10);
estim_params_.corrx = zeros(0, 11);
estim_params_.corrn = zeros(0, 11);
estim_params_.param_vals = zeros(0, 10);
estim_params_.var_exo = [estim_params_.var_exo; 1, NaN, 1e-20, 10, 4, 0.1, 2, NaN, NaN, NaN ];
estim_params_.var_exo = [estim_params_.var_exo; 2, NaN, 1e-20, 10, 4, 0.1, 2, NaN, NaN, NaN ];
estim_params_.var_exo = [estim_params_.var_exo; 3, NaN, 1e-20, 10, 4, 0.1, 2, NaN, NaN, NaN ];
estim_params_.var_exo = [estim_params_.var_exo; 4, NaN, 1e-20, 10, 4, 0.1, 2, NaN, NaN, NaN ];
estim_params_.var_exo = [estim_params_.var_exo; 5, NaN, 1e-20, 10, 4, 0.1, 2, NaN, NaN, NaN ];
estim_params_.var_exo = [estim_params_.var_exo; 6, NaN, 1e-20, 10, 4, 0.1, 2, NaN, NaN, NaN ];
estim_params_.var_exo = [estim_params_.var_exo; 7, NaN, 1e-20, 10, 4, 0.1, 2, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 29, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 31, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 32, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 34, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 35, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 36, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 37, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 8, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 7, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 11, NaN, 1e-20, 20, 3, 4, 4.5, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 13, NaN, 1e-20, 10, 3, 1.50, 1.11, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 14, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 19, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 22, NaN, 1e-20, 10, 3, 2, 2.25, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 21, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 18, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 20, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 10, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 17, NaN, 1e-20, 10, 3, 1.25, 0.36, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 25, NaN, 1, 10, 3, 1.5, 0.75, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 28, NaN, 1e-20, 0.9999, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 27, NaN, 1e-20, 1, 3, 0.12, 0.15, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 26, NaN, 1e-20, 1, 3, 0.12, 0.15, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 5, NaN, (-5), 5, 2, 0.625, 0.3, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 6, NaN, 1e-20, 2, 2, 0.25, 0.3, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 4, NaN, (-30), 30, 3, 0, 6, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 38, NaN, 1e-20, 1, 3, 0.4, 0.30, NaN, NaN, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 2, NaN, 1e-20, 1, 5, NaN, NaN, 0, 1, NaN ];
estim_params_.param_vals = [estim_params_.param_vals; 9, NaN, 1e-20, 1, 3, 0.3, 0.15, NaN, NaN, NaN ];
set_dynare_seed('clock');
options_.mh_replic = 0;
options_.mode_compute = 0;
options_.nodiagnostic = true;
options_.nograph = true;
options_.presample = 4;
options_.datafile = 'usdata_sw_mat';
options_.order = 1;
var_list_ = {};
oo_recursive_=dynare_estimation(var_list_);


oo_.time = toc(tic0);
disp(['Total computing time : ' dynsec2hms(oo_.time) ]);
if ~exist([M_.dname filesep 'Output'],'dir')
    mkdir(M_.dname,'Output');
end
save([M_.dname filesep 'Output' filesep 'Smets_Wouters_2007_45_results.mat'], 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'Smets_Wouters_2007_45_results.mat'], 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'Smets_Wouters_2007_45_results.mat'], 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'Smets_Wouters_2007_45_results.mat'], 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'Smets_Wouters_2007_45_results.mat'], 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'Smets_Wouters_2007_45_results.mat'], 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'Smets_Wouters_2007_45_results.mat'], 'oo_recursive_', '-append');
end
disp('Note: 33 warning(s) encountered in the preprocessor')
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
