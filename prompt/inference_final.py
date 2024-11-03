import sys
import os
real_dir=os.getcwd()
sys.path.append(real_dir)



from generator.predictor_final import predictor
predictor(#input_dir="G:/云南巧家宽频带数据/hdf5_merged",
       input_hdf5="g:/DiTing2.0/diting2.0_publish_desensitization_cenc/DiTing_2020_2021_desensitization.hdf5",
       input_model='test_trainer_outputs80/models/test_trainer_080.h5',
       output_dir='PredictionResults',
       detection_threshold=0.3,                
       P_threshold=0.1,
       S_threshold=0.1, 
       number_of_plots=1000,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(20000, 3),
       normalization_mode='std',
       mode='generator',
       number_of_cpus=4,
       batch_size=200,
       gpuid=0,
       gpu_limit=0.98)