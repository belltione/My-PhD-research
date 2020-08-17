function result = Times_series_EEGimage_generation(inputdata,outputpath,image_filename)   
%   Times_series_EEG_image = (1./(1+1*exp(-0.5*(inputdata))));  
  mkdir(outputpath);
  imwrite(inputdata, [outputpath '\' image_filename]);
  return
end