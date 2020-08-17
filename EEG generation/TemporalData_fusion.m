function final_EEG_image = TemporalData_fusion(EEGimageData,GRID_SCALE,frame_cooeficient) 
   final_EEG_image = zeros(GRID_SCALE,GRID_SCALE,3);
   
   for T=1:size(frame_cooeficient,2)       
       final_EEG_image = final_EEG_image + frame_cooeficient(T).*EEGimageData(:,:,:,T);        
   end
   
%    final_EEG_image(:,:,2) - frame_cooeficient(1).*EEGimageData(:,:,2,1) - frame_cooeficient(2).*EEGimageData(:,:,2,2) - frame_cooeficient(3).*EEGimageData(:,:,2,3) - frame_cooeficient(4).*EEGimageData(:,:,2,4) - frame_cooeficient(5).*EEGimageData(:,:,2,5)  
   return
end