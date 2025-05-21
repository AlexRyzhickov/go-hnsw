package embeddings

import "sync"

const RowLength = 128

type Matrix struct {
	data   []float32
	length int
	Mu     sync.RWMutex
}

func New(limit int) *Matrix {
	return &Matrix{
		data:   make([]float32, limit*RowLength),
		length: 0,
		Mu:     sync.RWMutex{},
	}
}

func (emb *Matrix) GetRow(pos int) []float32 {
	beg := RowLength * pos
	end := RowLength * (pos + 1)
	return emb.data[beg:end]
}

func (emb *Matrix) CopyRow(dest []float32, pos int) {
	beg := RowLength * pos
	end := RowLength * (pos + 1)
	emb.Mu.RLock()
	copy(dest, emb.data[beg:end])
	emb.Mu.RUnlock()
}

func (emb *Matrix) SetRow(vec []float32, pos int) {
	emb.grow(pos + 1)
	dest := emb.GetRow(pos)
	emb.Mu.Lock()
	copy(dest, vec)
	emb.Mu.Unlock()
}

func (emb *Matrix) grow(length int) {
	emb.Mu.RLock()
	ok := length <= emb.length
	emb.Mu.RUnlock()
	if ok {
		return
	}
	emb.Mu.Lock()
	if length > emb.length {
		emb.length = length
	}
	emb.Mu.Unlock()
}
