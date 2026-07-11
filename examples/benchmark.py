import time
import numpy as np
from mini_vectordb.store import VectorStore

def benchmark_search(n: int, dim: int=128, k: int=5)-> float:
    store = VectorStore(dim=dim)

    vectors = np.random.rand(n, dim)
    for i in range(n):
        store.add(id=f"vec_{i}", vector=vectors[i])
    
    query = np.random.rand(dim)

    start = time.perf_counter()
    store.search(query,k)
    end = time.perf_counter()

    return end - start

def main():
    sizes = [1_000, 10_000, 50_000, 100_000]

    print(f"{'n (records)':<15}{'search time (s)':<20}")
    print("-" * 35)
    for n in sizes:
        elapsed = benchmark_search(n)
        print(f"{n:<15}{elapsed:.4f}")

if __name__ == "__main__":
    main()