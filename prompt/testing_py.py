import sys
import os
real_dir=os.getcwd()
print(real_dir)
#sys.path.append('/nas-alinlp/xiaozhou.zyx/')
sys.path.append(real_dir)
print(sys.path)


from generator.tester import tester
tester(
       input_hdf5='h:/DiTing330km_part_0.hdf5',
       input_csv='h:/DiTing330km_part_0.csv',
       input_testset='test_trainer_outputs_tmp/test.npy',
       input_model='test_trainer_outputs80/models/test_trainer_080.h5',
       output_name='TestResults',             
       number_of_plots=1000,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(20000, 3),
       batch_size=200,
       gpuid=0,
       gpu_limit=0.98)