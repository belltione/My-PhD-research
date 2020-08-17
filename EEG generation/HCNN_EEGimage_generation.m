function result = HCNN_EEGimage_generation(Pixel_R,Pixel_G,Pixel_B,outputpath,image_filename)   
  
  Coor_X = [1,1,4,4,4,4,4,7,7,7,7,7,10,10,10,10,10,13,13,13,13,13,16,16,16,16,16,19,19,19];
  Coor_Y = [6,14,2,6,10,14,18,2,6,10,14,18,2,6,10,14,18,2,6,10,14,18,2,6,10,14,18,6,10,14];
  HCNN_EEG_image_R = zeros(20,20);
  HCNN_EEG_image_G = zeros(20,20);
  HCNN_EEG_image_B = zeros(20,20);
  
  Pixel_R = (1./(1+1*exp(-0.5*(Pixel_R))));
  Pixel_G = (1./(1+1*exp(-0.5*(Pixel_G))));
  Pixel_B = (1./(1+1*exp(-0.5*(Pixel_B))));
  % Put the power spectrum of EEG data to the corredponding location of the
  % output EEG image matrix.
  for i=1:30
      HCNN_EEG_image_R(Coor_X(1,i),Coor_Y(1,i)) = Pixel_R(1,i);
      HCNN_EEG_image_G(Coor_X(1,i),Coor_Y(1,i)) = Pixel_G(1,i);
      HCNN_EEG_image_B(Coor_X(1,i),Coor_Y(1,i)) = Pixel_B(1,i);
  end
 
  outputpath_R = [outputpath '_R'];
  mkdir(outputpath_R);
  imwrite(HCNN_EEG_image_R, [outputpath_R '\' image_filename]);
  outputpath_G = [outputpath '_G'];
  mkdir(outputpath_G);
  imwrite(HCNN_EEG_image_G, [outputpath_G '\' image_filename]);
  outputpath_B = [outputpath '_B'];
  mkdir(outputpath_B);
  imwrite(HCNN_EEG_image_B, [outputpath_B '\' image_filename]);
  return
end