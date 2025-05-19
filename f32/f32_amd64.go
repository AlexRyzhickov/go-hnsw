//go:build !noasm && !appengine
// +build !noasm,!appengine

package f32

import (
	"gonum.org/v1/gonum/blas/blas32"
)

func L2Squared(x, y []float32) float32

func L2Squared8AVX(x, y []float32) float32

func InnerProduct(x, y []float32) float32 {
	if len(x) != len(y) {
		panic("len(x) != len(y)")
	}

	xVec := blas32.Vector{
		N:    len(x),
		Inc:  1,
		Data: x,
	}

	yVec := blas32.Vector{
		N:    len(y),
		Inc:  1,
		Data: y,
	}

	return 1 - blas32.Dot(xVec, yVec)
}
