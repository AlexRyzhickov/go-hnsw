package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	hnsw "go-hnsw"
	"go-hnsw/embeddings"

	usearch "github.com/unum-cloud/usearch/golang"
)

func TestGoHnsw(t *testing.T) {
	const (
		M              = 32
		efConstruction = 400
		efSearch       = 100
		K              = 1000
	)
	const (
		queriesSize = 1000
		pointsSize  = 100_000
	)

	matrix, err := matrixFromFile(pointsSize, "vectors.txt")
	if err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	h := hnsw.New(M, efConstruction, matrix.GetRow(0))
	h.Grow(pointsSize)
	for i := 1; i < pointsSize; i++ {
		point := matrix.GetRow(i)
		h.Add(point, uint32(i))
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}
	indexBuildTime := time.Since(start)

	fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")
	queries := generateQueries(queriesSize)
	gt := generateGT(matrix, queries, pointsSize, queriesSize, K)
	fmt.Printf("Now searching with HNSW...\n")

	recall := float64(0)
	for i := 0; i < queriesSize; i++ {
		result := h.Search(queries[i], efSearch, K)
		our := make([]candidate, K)
		for j := 0; j < K; j++ {
			item := result.Pop()
			if item == nil {
				break
			}
			our[j] = candidate{id: item.ID, score: item.D}
		}
		recall = recall + Recall(gt[i], our)
	}

	fmt.Printf("Hnsw index building time: %v\n", indexBuildTime)
	fmt.Printf("Recall: %v\n", float64(recall)/float64(queriesSize))
}

func TestUsearch(t *testing.T) {
	const (
		M              = 32
		efConstruction = 400
		efSearch       = 100
		K              = 1000
	)
	const (
		queriesSize = 1000
		pointsSize  = 100_000
	)

	matrix, err := matrixFromFile(pointsSize, "vectors.txt")
	if err != nil {
		t.Fatal(err)
	}

	conf := usearch.IndexConfig{
		Quantization:    usearch.F32,
		Metric:          usearch.InnerProduct,
		Dimensions:      uint(128),
		Connectivity:    M,
		ExpansionAdd:    efConstruction,
		ExpansionSearch: efSearch,
		Multi:           false,
	}

	start := time.Now()
	index, err := usearch.NewIndex(conf)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Destroy()
	err = index.Reserve(uint(pointsSize))
	if err != nil {
		log.Fatal(err)
	}
	for i := 0; i < pointsSize; i++ {
		point := matrix.GetRow(i)
		err := index.Add(usearch.Key(i), point)
		if err != nil {
			t.Fatal(err)
		}
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}
	indexBuildTime := time.Since(start)

	fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")
	queries := generateQueries(queriesSize)
	gt := generateGT(matrix, queries, pointsSize, queriesSize, K)
	fmt.Printf("Now searching with HNSW...\n")

	recall := float64(0)
	for i := 0; i < queriesSize; i++ {
		keys, distances, err := index.Search(queries[i], K)
		if err != nil {
			t.Fatal(err)
		}

		our := make([]candidate, K)
		for j := 0; j < K; j++ {
			our[j] = candidate{id: uint32(keys[j]), score: distances[j]}
		}
		recall = recall + Recall(gt[i], our)
	}

	fmt.Printf("Hnsw index building time: %v\n", indexBuildTime)
	fmt.Printf("Recall: %v\n", float64(recall)/float64(queriesSize))
}

func matrixFromFile(limit int, path string) (*embeddings.Matrix, error) {
	matrix := embeddings.New(limit)

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	count := 0
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)

		vec := make([]float32, len(fields))
		for i, f := range fields {
			val, err := strconv.ParseFloat(f, 32)
			if err != nil {
				return nil, err
			}
			vec[i] = float32(val)
		}
		matrix.SetRow(vec, count)
		count++
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return matrix, nil
}

func TestGen(t *testing.T) {
	const (
		pointsSize = 100_000
	)
	file, err := os.Create("vectors.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	for i := 0; i < pointsSize; i++ {
		vec := randomPoint()
		for j, v := range vec {
			if j > 0 {
				_, err = file.WriteString(" ")
				if err != nil {
					t.Fatal(err)
				}
			}
			_, err = file.WriteString(fmt.Sprintf("%f", v))
			if err != nil {
				t.Fatal(err)
			}
		}
		_, err = file.WriteString("\n")
		if err != nil {
			t.Fatal(err)
		}
	}
}
