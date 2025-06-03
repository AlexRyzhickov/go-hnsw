package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRecall(t *testing.T) {
	type test struct {
		gt     []candidate
		our    []candidate
		recall float64
	}
	tests := []test{
		{
			gt: []candidate{
				{id: 0, score: 1.0},
				{id: 1, score: 1.0},
				{id: 7, score: 1.1},
				{id: 8, score: 1.2},
				{id: 9, score: 1.2},
				{id: 4, score: 1.3},
				{id: 5, score: 1.4},
				{id: 6, score: 1.4},
			},
			our: []candidate{
				{id: 0, score: 1.0},
				{id: 1, score: 1.0},
				{id: 7, score: 1.1},
				{id: 8, score: 1.2},
				{id: 9, score: 1.2},
				{id: 4, score: 1.3},
				{id: 5, score: 1.4},
				{id: 6, score: 1.4},
			},
			recall: float64(8) / float64(8),
		},
		{
			gt: []candidate{
				{id: 0, score: 1.0},
				{id: 1, score: 1.0},
				{id: 7, score: 1.1},
				{id: 8, score: 1.2},
				{id: 9, score: 1.2},
				{id: 4, score: 1.3},
				{id: 5, score: 1.4},
				{id: 6, score: 1.4},
			},
			our: []candidate{
				{id: 0, score: 1.0},
				{id: 1, score: 1.0},
				{id: 7, score: 1.1},
				{id: 8, score: 1.2},
				{id: 9, score: 1.2},
				{id: 2, score: 1.4},
				{id: 5, score: 1.4},
				{id: 6, score: 1.4},
			},
			recall: float64(7) / float64(8),
		},
		{
			gt: []candidate{
				{id: 0, score: 1.0},
				{id: 1, score: 1.0},
				{id: 7, score: 1.1},
				{id: 8, score: 1.2},
				{id: 9, score: 1.2},
				{id: 4, score: 1.3},
				{id: 5, score: 1.4},
				{id: 6, score: 1.4},
			},
			our: []candidate{
				{id: 0, score: 1.0},
				{id: 1, score: 1.0},
				{id: 7, score: 1.1},
				{id: 2, score: 1.4},
				{id: 5, score: 1.4},
				{id: 6, score: 1.4},
				{id: 10, score: 1.5},
				{id: 11, score: 1.5},
			},
			recall: float64(5) / float64(8),
		},
		{
			gt: []candidate{
				{id: 0, score: 0.0},
				{id: 1, score: 1.0},
				{id: 2, score: 2.0},
			},
			our: []candidate{
				{id: 1, score: 1.0},
			},
			recall: float64(1) / float64(3),
		},
	}

	for _, tc := range tests {
		recall := Recall(tc.gt, tc.our)
		assert.Equal(t, tc.recall, recall)
	}
}
