function [par] = Cal_par(prob_k)
% CEC2020 Constrained Optimization Test Suite 
% Abhishek Kumar (email: abhishek.kumar.eee13@iitbhu.ac.in, Indian Institute of Technology (BHU), Varanasi) 

% prob_k -> Index of problem.
% par.n  -> Dimension of the problem.
% par.g  -> Number of inequility constraints.
% par.h  -> Number of equality constraints.
% par.xmin -> lower bound of decision variables.
% par.xmax -> upper bound of decision variables.


D        = [9	11	7	6	9	38	48	2	3	3	7	7	5	10	7	14	3	4	4	2	5	9	5	7	4	22	10	10	4	3	4	5	...
            30	118	153	158	126	126	126	76	74	86	86	30	25	25	25	30	30	30	59	59	59	59	64	64	64];
par.n    = D(prob_k);     
gn       = [0,0,14,1,2,0,0,2,1,3,4,9,3,10,11,15,3,4,5,3,7,10,8,7,7,86,3,9,1,8,1,6,30,0,0,0,0,0,0,0,0,0,0,105,24,24,24,29,29,29,14,14,14,14,0,0,0];
hn       = [8,9,0,4,4,32,38,0,1,0,4,0,0,0,0,0,0,0,0,0,0,1,3,0,0,0,0,0,0,0,1,0,0,108,148,148,116,116,116,76,74,76,76,0,1,1,1,1,1,1,1,1,1,1,6,6,6];
par.gn    = gn;
par.hn    = hn;
par.g     = gn(prob_k);
par.h     = hn(prob_k);
%% range
% bound constraint definitions for all 18 test functions
xmin1    = [0,0,0,0,1000,0,100,100,100];
xmax1    = [10,200,100,200,2000000,600,600,600,900];
xmin2    = [10^4,10^4,10^4,0,0,0,100,100,100,100,100];
xmax2    = [0.819*10^6, 1.131*10^6, 2.05*10^6,0.05074,0.05074,0.05074,200,300,300,300,400];
xmin3    = [1000,0,2000,0,0,0,0];
xmax3    = [2000,100,4000,100,100,20,200];
xmin4    = [0,0,0,0,1e-5,1e-5];
xmax4    = [1,1,1,1,16,16];
xmin5    = -0*ones(1,par.n);
xmax5    = [100,200,100,100,100,100,200,100,200];
xmin6    = 0*ones(1,par.n);
xmax6    = [90,150,90,150,90,90,150,90,90,90,150,150,90,90,150,90,150,90,150,90,1,1.2,1,1,1,0.5,1,1,0.5,0.5,0.5,1.2,0.5,1.2,1.2,0.5,1.2,1.2];
xmin7    = -0*ones(1,par.n); xmin7([24,26,28,31]) = 0.849999;
xmax7    = 1*ones(1,par.n); xmax7(4) = 140; xmax7([25,27,32,35,37,29]) = 30;xmax7([2,3,5,13,14,15]) = 90; xmax7([1,6,7,8,9,10,11,12,16,17,18,19,20]) = 35;
xmin8    = [0,-0.51];
xmax8    = [1.6,1.49];
xmin9    = [0.5,0.5,-0.51];
xmax9    = [1.4,1.4,1.49];
xmin10   = [0.2, -2.22554, -0.51];
xmax10   = [1, -1, 1.49];
xmin11   = [0,0,0,0,-0.51,-0.51,0];
xmax11   = [20,20,10,10,1.49,1.49,40];
xmin12   = [0,0,0,-0.51,-0.51,-0.51,-0.51];
xmax12   = [100,100,100,1.49,1.49,1.49,1.49]; 
xmin13   = [27,27,27,77.51,32.51];
xmax13   = [45,45,45,102.49,45.49];
xmin14   = [ 0.51,0.51,0.51,250,250,250,6,4,40,10];
xmax14   = [3.49,3.49,3.49,2500,2500,2500,20,16,700,450];
xmin15   = [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5];
xmax15   = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5];
xmin16   = 0.001*ones(1,par.n);
xmax16   = +5*ones(1,par.n);
xmin17   = [0.05,0.25,2.00];
xmax17   = [2,1.3,15.0];
xmin18   = [0.51,0.51,10,10];
xmax18   = [99.49,99.49,200,200];
xmin19   = [0.125,0.1,0.1,0.1];
xmax19   = [2,10,10,2];
xmin20   = 0*ones(1,par.n);
xmax20   = 1*ones(1,par.n); 
xmin21   = [60,90,1,0,2];
xmax21   = [80,110,3,1000,9];
xmin22   = [16.51,13.51,13.51,16.51,13.51,47.51,0.51,0.51,0.51];
xmax22   = [96.49,54.49,51.49,46.49,51.49,124.49,3.49,6.49,6.49];
xmin23   = [0,0,0,0,0];
xmax23   = [60,60,90,90,90];
xmin24   = [10,10,100,0,10,100,1];
xmax24   = [150,150,200,50,150,300,3.14];
xmin25   = [ 1, 1,  1e-6,1];
xmax25   = [16, 16, 16*1e-6,16];
xmin26   = [ 6.51.*ones(1,8), 0.51.*ones(1,14)];
xmax26   = [ 76.49.*ones(1,8), 4.49.*ones(1,4), 9.49.*ones(1,10)];
xmin27   = 0.645e-4*ones(1,par.n);
xmax27   = 50e-4*ones(1,par.n); 
xmin28   = [125,10.5,4.51,0.515,0.515,0.4,0.6,0.3,0.02,0.6];
xmax28   = [150,31.5,50.49,0.6,0.6,0.5,0.7,0.4,0.1,0.85];
xmin29   = [20,1,20,0.1];
xmax29   = [50,10,50,60];
xmin30   = [0.51,0.6,0.51];
xmax30   = [70.49,3,42.49];
xmin31   = 12.*ones(1,4);
xmax31   = 60.*ones(1,4);
xmin32   = [78,33,27,27,27];
xmax32   = [102,45,45,45,45];
xmin33   = 0.001.*ones(1,par.n);
xmax33   = ones(1,par.n);
xmin34   = -1*ones(1,par.n);
xmax34   = +1*ones(1,par.n);
xmin35   = -1*ones(1,par.n);
xmax35   = +1*ones(1,par.n);
xmin36   = -1*ones(1,par.n);
xmax36   = +1*ones(1,par.n);
xmin37   = -1*ones(1,par.n);xmin37(117:126) = 0;
xmax37   = +1*ones(1,par.n);
xmin38   = -1*ones(1,par.n);xmin38(117:126) = 0;
xmax38   = +1*ones(1,par.n);
xmin39   = -1*ones(1,par.n);xmin39(117:126) = 0;
xmax39   = +1*ones(1,par.n);
xmin40   = -1*ones(1,par.n);xmin40(75:76) = 0;
xmax40   = +1*ones(1,par.n);xmax40(75:76) = 2;
xmin41   = -1*ones(1,par.n);
xmax41   = +1*ones(1,par.n);
xmin42   = -1*ones(1,par.n);xmin42(75:76) = 0;xmin42(77:86) = 0;
xmax42   = +1*ones(1,par.n);xmax42(75:76) = 2;xmax42(77:86) = 500;
xmin43   = -1*ones(1,par.n);xmin43(75:76) = 0;xmin43(77:86) = 0;
xmax43   = +1*ones(1,par.n);xmax43(75:76) = 2;xmax43(77:86) = 500;
xmin44   = 40*ones(1,par.n);
xmax44   = 1960*ones(1,par.n);
xmin45   = -0*ones(1,par.n);
xmax45   = +90*ones(1,par.n);
xmin46   = -0*ones(1,par.n);
xmax46   = +90*ones(1,par.n);
xmin47   = -0*ones(1,par.n);
xmax47   = +90*ones(1,par.n);
xmin48   = -0*ones(1,par.n);
xmax48   = +90*ones(1,par.n);
xmin49   = -0*ones(1,par.n);
xmax49   = +90*ones(1,par.n);
xmin50   = -0*ones(1,par.n);
xmax50   = +90*ones(1,par.n);
xmin51   = 0.*ones(1,par.n);
xmax51   = 10.*ones(1,par.n);
xmin52   = 0.*ones(1,par.n);
xmax52   = 10.*ones(1,par.n);
xmin53   = 0.*ones(1,par.n);
xmax53   = 10.*ones(1,par.n);
xmin54   = 0.*ones(1,par.n);
xmax54   = 10.*ones(1,par.n);
xmin55   = 0.*ones(1,par.n);
xmax55   = 10.*ones(1,par.n);
xmin56   = 0.*ones(1,par.n);
xmax56   = 10.*ones(1,par.n);
xmin57   = 0.*ones(1,par.n);
xmax57   = 10.*ones(1,par.n);

eval(['par.xmin=xmin' int2str(prob_k) ';']);
eval(['par.xmax=xmax' int2str(prob_k) ';' ]);
end