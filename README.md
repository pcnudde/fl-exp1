# fl-exp1

Simple exploration of nvflare. Use trivial MNIST training as example

## Files

**train_0.py** 
- Minimal MNIST training script
- 37 loc
- Simple 2-layer fully-connected network (784 → 128 → 10)
- Achieves 97.47% accuracy

**train_1.py**
- Federated learning simulation with 3 sites
- Splits MNIST dataset into 3 equal parts (one per site)
- Runs 5 federated learning rounds (FedAvg)
- Achieves 96.35% accuracy
- 56 loc

**train_2.py**
- as above but use multiprocessing to run sites in parallel
- 64 loc



