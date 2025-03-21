void vector_add(const float* A, const float* B, float* C, int N) {
  for (int i = 0; i < N; i++) { C[i] = A[i] + B[i]; }
}

// A, B, C are pointers (i.e. pointers to memory on the CPU)
void solve(const float* A, const float* B, float* C, int N) { vector_add(A, B, C, N); }
