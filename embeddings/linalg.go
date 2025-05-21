package embeddings

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

func (emb *Matrix) CalcDistances(dest []float32, vec []float32) []float32 {
	emb.Mu.RLock()
	defer emb.Mu.RUnlock()

	if cap(dest) < emb.length {
		dest = make([]float32, emb.length)
	}
	dest = dest[:emb.length]

	A := blas32.General{
		Rows:   emb.length,
		Cols:   RowLength,
		Stride: RowLength,
		Data:   emb.data,
	}
	x := vector(vec)
	y := vector(dest)
	blas32.Gemv(blas.NoTrans, 1, A, x, 0, y)
	return dest
}

func vector(vec []float32) blas32.Vector {
	return blas32.Vector{
		N:    len(vec),
		Inc:  1,
		Data: vec,
	}
}
