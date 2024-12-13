import numpy as np
from uniform_instance_gen import uni_instance_gen

test_instance_sizes = [(6, 6), (10, 10), (15, 10), (15, 15), (20, 10), (20, 15), (20, 20), (30, 15), (30, 20),
                       (50, 20), (100, 20), (200, 50)]
# j = 20
# m = 10
l = 1
h = 99
batch_size = 100
seed = 300
for j, m in test_instance_sizes:
    print(j, m)
    np.random.seed(seed)

    data = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
    print(data.shape)
    np.save('./test_generated_data/generatedData{}_{}_Seed{}.npy'.format(j, m, seed), data)