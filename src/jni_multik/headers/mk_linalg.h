#ifndef CPP_CPP_LINALG_H_
#define CPP_CPP_LINALG_H_

extern "C" {
// svd
extern void dgesvd_(char const *, char const *, int const *, int const *, double *, int const *, double *,
                    double *, int const *, double *, int const *, double *, int const *, int *);

// norm
extern double dlange_(char const *, int const *, int const *, double const *, int const *, double *);

// eigenvalues
extern void dgeev_(char const *, char const *, int const *, double *, int const *, double *, double *,
                   double *, int const *, double *, int const *, double *, int const *, int *);

// dot matrix
extern void dgemm_(char *, char *, int *, int *, int *, double *,
                   double *, int *, double *, int *, double *, double *, int *);

}
void matrix_power() {

}

void svd() {

}

void norm() {

}

void det() {

}

void matrix_rank() {

}

void matrix_solve() {

}

void inv() {

}

void eigen_values(double *A, int n) {
  const char Nchar = 'N';
  double *eigReal = new double[n];
  double *eigImag = new double[n];
  const int lwork = 5 * n;
  double *work = new double[lwork];
  int info;
  double *lv, *rv;
  const int one = 1;
// перезаписывает матрицу
  dgeev_(&Nchar, &Nchar, &n, A, &n, eigReal, eigImag, nullptr, &one, nullptr, &one, work, &lwork, &info);

  delete [] eigReal;
  delete [] eigImag;
  delete [] work;
}

void matrix_dot(double *A, int n, int m, int k, double *B, double *C) {
  char Nchar = 'N';
  double alpha = 1.0;
  dgemm_(&Nchar, &Nchar, &n, &m, &k, &alpha, A, &n, B, &k, &alpha, C, &n);
}

#endif //CPP_CPP_LINALG_H_
