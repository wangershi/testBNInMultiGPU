# testBNInMultiGPU
test Batch Normalization in multi GPU
syncBatchNorm in validSyncBN.py is most robust.

## result
testBNInSingleGPU():
```Python
outputArray = [0. 0. 0.]

After normalization:
moving mean = [0.1 0.1 0.1]
moving variance = [0.9 0.9 0.9]
```

testBNInMultiGPU1():
```Python
outputArray = [0. 0. 0.]

After normalization:
moving mean = [0.2 0.2 0.2]
moving variance = [0.79999995 0.79999995 0.79999995]
```

testBNInMultiGPU2():
```Python
outputArray = [0. 0. 0.]

After normalization:
moving mean = [0.3 0.3 0.3]
moving variance = [0.79999995 0.79999995 0.79999995]
```

## blog writen in Chinese
[[TensorFlow]Batch Normalization在单机多卡上的一些实验](https://zhuanlan.zhihu.com/p/69267784)
