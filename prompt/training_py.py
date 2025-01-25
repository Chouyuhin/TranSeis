import sys
import os
real_dir=os.getcwd()
#print(real_dir)
sys.path.append(real_dir)
#print(sys.path)


from generator.trainer import trainer
trainer(input_dir='G:/DiTing_datas/',
        num_part=1,
        output_name='TrainResults',                
        cnn_blocks=2,
        lstm_blocks=1,
        padding='same',
        activation='relu',
        drop_rate=0.2,
        shift_event_r=0.9,
        loss_weights=[0.2, 0.3, 0.5],
        train_valid_test_split=[0.60, 0.20, 0.20],
        batch_size=200,
        epochs=80, 
        patience=12,
        gpuid=0,
        gpu_limit=0.98)
