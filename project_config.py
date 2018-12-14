import os

base_dir =  "H:/data/luna2016/"
annatation_file = base_dir + 'CSVFILES/annotations.csv'
candidate_file =base_dir +  'CSVFILES/candidates_V2.csv'
plot_output_path = base_dir + 'cubic_npy'
if not os.path.exists(plot_output_path):
    os.mkdir(plot_output_path)
normalazation_output_path = base_dir +'cubic_normalization_npy'
if not os.path.exists(normalazation_output_path):
    os.mkdir(normalazation_output_path)
test_path = base_dir + 'cubic_normalization_test/'
if not os.path.exists(test_path):
    os.mkdir(test_path)
###  training and test configuration #####
batch_size = 32
learning_rate = 0.01
keep_prob = 0.7