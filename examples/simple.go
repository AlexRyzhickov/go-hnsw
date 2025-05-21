package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	hnsw "go-hnsw"
	"go-hnsw/embeddings"
)

func main() {

	const (
		M              = 32
		efConstruction = 400
		efSearch       = 100
		K              = 100
	)
	const (
		queriesSize = 1000
		pointsSize  = 10_000
	)

	matrix := embeddings.New(pointsSize)

	var zero hnsw.Point = make([]float32, 128)

	h := hnsw.New(M, efConstruction, zero)
	h.Grow(pointsSize)
	matrix.SetRow(zero, 0)
	for i := 1; i < pointsSize; i++ {
		point := randomPoint()
		matrix.SetRow(point, i)
		h.Add(point, uint32(i))
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}

	fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")
	queries := make([]hnsw.Point, queriesSize)

	gt := make([][]uint32, queriesSize)
	for i := range queries {
		queries[i] = randomPoint()
	}
	for i := range queries {
		dest := make([]float32, pointsSize)
		indices := make([]uint32, pointsSize)
		for j := 0; j < pointsSize; j++ {
			indices[j] = uint32(j)
		}
		dest = matrix.CalcDistances(dest, queries[i])
		for j := 0; j < pointsSize; j++ {
			dest[j] = 1 - dest[j]
		}
		sort.Slice(indices, func(i, j int) bool { return dest[indices[i]] < dest[indices[j]] })
		indices = indices[:K]
		sort.Slice(indices, func(i, j int) bool { return dest[indices[i]] > dest[indices[j]] })
		gt[i] = make([]uint32, K)
		for k := 0; k < K; k++ {
			gt[i][k] = indices[k]
		}
	}

	fmt.Printf("Now searching with HNSW...\n")

	recall := float32(0)
	start := time.Now()
	for i := 0; i < queriesSize; i++ {
		result := h.Search(queries[i], efSearch, K)
		hits := 0
		items := make(map[uint32]struct{})
		for j := 0; j < K; j++ {
			item := result.Pop()
			items[item.ID] = struct{}{}
		}
		for j := 0; j < K; j++ {
			id := gt[i][j]
			if _, ok := items[id]; ok {
				hits++
			}
		}
		recall = recall + float32(hits)/float32(K)
	}
	stop := time.Since(start)

	fmt.Printf("%v queries / second (single thread)\n", 1000.0/stop.Seconds())
	fmt.Printf("Average 10-NN precision: %v\n", float64(recall)/float64(queriesSize))
}

func randomPoint() hnsw.Point {
	sum := float32(0)
	var v hnsw.Point = make([]float32, 128)
	for i := 0; i < len(v); i++ {
		v[i] = 2*rand.Float32() - 1
		sum = sum + v[i]*v[i]
	}
	norm := float32(math.Sqrt(float64(sum)))
	for i := 0; i < len(v); i++ {
		v[i] = v[i] / norm
	}
	return v
}

//queriesSize = 1000
//pointsSize  = 100_000
//h.DistFunc = f32.InnerProduct
//Average 10-NN precision: 0.90825

//queriesSize = 1000
//pointsSize  = 100_000
//h.DistFunc = f32.L2Squared8AVX
//Average 10-NN precision:  0.53306

//queriesSize = 1000
//pointsSize  = 1_000_000
//h.DistFunc = f32.InnerProduct
//Average 10-NN precision: 0.87178
