package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	hnsw "go-hnsw"
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
		pointsSize  = 100_000
	)

	var zero hnsw.Point = make([]float32, 128)

	h := hnsw.New(M, efConstruction, zero)
	//h.Grow(size)
	h.Grow(pointsSize)

	//matrix := make([][]float32, 0)

	for i := 1; i <= pointsSize; i++ {
		point := randomPoint()
		//matrix = append(matrix, point)
		h.Add(point, uint32(i))
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}

	fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")
	queries := make([]hnsw.Point, queriesSize)
	truth := make([][]uint32, queriesSize)
	//truth := make([][]float32, 1000)
	for i := range queries {
		queries[i] = randomPoint()
		result := h.SearchBrute(queries[i], K)
		//truth[i] = make([]float32, K)
		truth[i] = make([]uint32, K)
		for j := K - 1; j >= 0; j-- {
			item := result.Pop()
			truth[i][j] = item.ID
		}

		//type SDoc struct {
		//	Id    uint32
		//	Score float32
		//}
		//
		//docs := make([]SDoc, 0)
		//for j := 0; j < size; j++ {
		//	score := f32.L2Squared8AVX(matrix[j], queries[i])
		//	docs = append(docs, SDoc{Id: uint32(j + 1), Score: score})
		//}
		//
		//sort.Slice(docs, func(i, j int) bool {
		//	return docs[i].Score < docs[j].Score
		//})
		//
		//docs = docs[:K]
		//
		//truth[i] = make([]float32, K)
		//for j := 0; j < K; j++ {
		//	truth[i][j] = docs[j].Score
		//}
	}

	fmt.Printf("Now searching with HNSW...\n")
	hits := 0
	start := time.Now()
	for i := 0; i < queriesSize; i++ {
		result := h.Search(queries[i], efSearch, K)
		for j := 0; j < K; j++ {
			item := result.Pop()
			for k := 0; k < K; k++ {
				if item.ID == truth[i][k] {
					hits++
				}
			}
		}
	}
	stop := time.Since(start)

	fmt.Printf("%v queries / second (single thread)\n", 1000.0/stop.Seconds())
	fmt.Printf("Average 10-NN precision: %v\n", float64(hits)/(1000.0*float64(K)))

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
