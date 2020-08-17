function result = FFT2Image(Inputdata,Inputchannel,transform_mode) 
   result = (1./(1+1*exp(-0.5*(Inputdata))));
   switch(transform_mode)
       case {'AlphaTheta_Gaussian'},
           if strcmp(Inputchannel,'alpha') == 1  && ((Inputdata(1,25) >= 5) || (Inputdata(1,26) >= 5))
               load('gaussian_weight.mat');
               result = result.* gaussian_weight;  
           end
           if strcmp(Inputchannel,'theta') == 1  && ((Inputdata(1,1) >= 5) || (Inputdata(1,2) >= 5))
               load('gaussian_weight.mat');
               result = result.* gaussian_weight;  
           end
       case {'AlphaGaussian'},
           if strcmp(Inputchannel,'alpha') == 1 
               load('gaussian_weight.mat');
               result = result.* gaussian_weight;  
           end
       case {'Alphafiltered'},    
           result = zeros(1,size(Inputdata,2));
           if strcmp(Inputchannel,'alpha') == 1 && ((Inputdata(1,25) >= 4) || (Inputdata(1,26) >= 4))
               for ch = 1:size(Inputdata,2)
                   if Inputdata(ch) <= -10
                       result(1,ch) = 0;
                   elseif Inputdata(ch) >= 10
                       result(1,ch) = 0.999999; % Assign 1 will get error (Dimension exceed of EEGchannel_statistic)
                   else
                       result(1,ch) = (Inputdata(ch) + 10) / 20;
                   end
               end
           else
               result = (1./(1+1*exp(-0.5*(Inputdata))));
           end
       case {'None'},

       otherwise
           error('[FFT2Image.m]: Wrong "transform_mode" input');
   end
   return;
end