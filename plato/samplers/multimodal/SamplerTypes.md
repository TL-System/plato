
There are several types of noniid samplers.


# label noniid - Label Distribution Skew

1. quantity-based label noniid, each client contains a fixed size of classes and each class contain the almost same amount of samples.
    quantity_label_noniid.py

2. distribution-based label noniid, each client contains all classes but with different distribution. The distribution of classes within
    one client follows the dirichlet distribution. 
    dirchlet.py

# sample noniid - Quantity Skew

1. quantity-based sample noniid, each client contains a different amount of samples. The sample amount distribution among clients follows  the dirichlet distribution. Each client contains all classes while the number of samples in each class is almost the same.
    sample_quantity_noniid.py


# distribution noniid - Label Distribution Skew + Quantity Skew
    distribution_noniid.py