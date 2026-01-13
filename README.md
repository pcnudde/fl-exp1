# fl-exp1

Simple exploration of nvflare. Use trivial MNIST training as example

## Files

**train_0.py** 
- Minimal MNIST training script
- 37 loc
- Achieves 97.47% accuracy

**train_1.py**
- Federated learning simulation with 3 sites
- Splits MNIST dataset into 3 equal parts (one per site)
- Runs 5 federated learning rounds (FedAvg)
- Achieves 96.35% accuracy
- 59 loc

**train_2.py**
- as above but use multiprocessing to run sites in parallel
- 67 loc

**train_3.py**
- use nvflare collab
- 69 loc
- much slower than multiprocess version
- only a few changes needed
   - use the collab decorators
   - get site-id from context instead of via function parameter
   - prep function call and unpack results slightly differently
   - use foxrecipy to start simulation





