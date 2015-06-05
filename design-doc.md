# SPARK-5992: Locality Sensitive Hashing (LSH) for MLlib

- Author: Yu Ishikawa
- Revision History:
    - initial version


## Overview

Locality-sensitive hashing (LSH) reduces the dimensionality of high-dimensional data. 
LSH hashes input items so that similar items map to the same “buckets” with high probability (the number of buckets being much smaller than the universe of possible input items). 
LSH differs from conventional and cryptographic hash functions because it aims to maximize the probability of a “collision” for similar items.
Locality-sensitive hashing has much in common with data clustering and nearest neighbor search.

## Requirements

There several major features we need to implement for the LSH API.

1. API consistency with Scala/Java/Python/(R).
2. Ease of Use.
3. Ease of Development/Extensibility.
4. Support `spark.ml`
4. Good performance.
5. Save/Load.

If we support algorithms which need a pre-calculation like a `train` method of a machine learning algorithm, we should support features for save and load for Spark Streaming.

## List of Algorithms

There are some basic LSH algorithms like below for Euclidean distance, Jaccard coefficnent and so on.

- Bit sampling for Hamming distance
- Min-wise independent permutations
- Nilsimsa Hash
- Random projection
- Min Hash

## Interfaces

It is pretty hard to define a common interface.
Because LSH algorithm has two types at least.
One is to calculate hash value. 
The other is to calculate a similarity between a feature(vector) and another one.

For example, random projection algorithm is a type of calculating a similarity.
It is designed to approximate the cosine distance between vectors.
On the other hand, min hash algorithm is a type of calculating a hash value.
The hash function maps a d dimensional vector onto a set of integers.

### Namespace

The new set of APIs will live under `org.apache.spark.mllib.feature`.

### Builder Pattern vs. Object Function

#### Builder Pattern Class

```
abstract class LSH {
  def getSimilarity(v1: Vector, v2: Vector): Double
}

abstract class MapTypeLSH[T] extends LSH {
  def map(v: Vector): T
}
```

```
class MinHash extends MapTypeLSH[Int] {
  override def getSimilarity(v1: Vector, v2: Vector): Double = ...

  override def map(v: Vector): Int = ...
}

val hash = new MinHash().setSeed(seed)
val hashValue = hash.map(v)
val similarity = hash.getSimilarity(v1, v2)
```

There are pros and cons with this design.

- Pros:
    - We can simplify how to call the LSH function.
    - We can hide complicated arguments. Because paramter tuning of some LSH is pretty haed.
- Cons:
    - It is a little bit complicated to create one because we should implement setters for the builder pattern.

#### Object function with arguments

```
val hashValue = MinHash.hash(v, seed)
val similarity = MinHash.compare(v1, v2, seed)
```

### Save/Load

If we support LSH algorithms which need pre-calculation, we should implement save and load functions.

```
## save a LSH which need pre-calculation.
val hash = new SavableHash().setX(x).setY(y)
hash.train(data)
hash.save(path)

## load the LSH.
val loadedHash = SavableHash.load(path)
```

## Reference

- [Locality Sensitive Hashing (LSH) Home Page](http://www.mit.edu/~andoni/LSH/)
- [Locality-sensitive hashing - Wikipedia, the free encyclopedia](http://en.wikipedia.org/wiki/Locality-sensitive_hashing)
