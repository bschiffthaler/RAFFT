#include <armadillo>
#include <cmath>
#include <iostream>

int64_t findMid(arma::vec const &spec) {
  int64_t m = 0;
  if (spec.n_elem % 2 == 0) {
    m = spec.n_elem / 2;
  } else {
    m = (spec.n_elem / 2) + 1;
  }
  m -= 1; // 0-based
  int64_t n = m / 4;
  arma::vec sv = spec.subvec(m - n , m + n);
  int64_t imin = sv.index_min();
  return imin + m - n;
}

arma::vec move(arma::vec const &seg, int64_t lag) {
  if (lag == 0 || lag >= static_cast<int64_t>(seg.n_elem)) {
    return seg;
  }
  if (lag > 0) {
    arma::vec ins(lag, arma::fill::ones);
    ins *= seg(0);
    arma::vec movedSeg =
        arma::join_cols(ins, seg.subvec(0, seg.n_elem - lag - 1));
    return movedSeg;
  } else {
    lag = std::abs(lag);
    // Fill insert with length lag with last element of segment
    arma::vec ins(lag, arma::fill::ones);
    ins *= seg(seg.n_elem - 1);
    // Join
    arma::vec movedSeg =
        arma::join_cols(seg.subvec(lag, seg.n_elem - 1), ins);
    return movedSeg;
  }
}

int64_t FFTcorr(arma::vec const &spectrum, arma::vec const &target,
                int64_t shift) {
  int64_t lag;
  int64_t m = target.n_elem;
  int64_t diff = 1000000;
  for (int64_t i = 1; i < 21; i++) {
    int64_t curdiff = std::pow(2, i);
    if (curdiff > m) {
      curdiff -= m;
      if (curdiff < diff) {
        diff = curdiff;
      }
    }
  }
  // std::cerr << "m -> " << m << ", diff -> " << diff << '\n';
  arma::vec target_cp(m + diff, arma::fill::zeros);
  arma::vec spectrum_cp(m + diff, arma::fill::zeros);
  for (uint64_t i = 0; i < target.n_elem; i++) {
    target_cp(i) = target(i);
  }
  for (uint64_t i = 0; i < spectrum.n_elem; i++) {
    spectrum_cp(i) = spectrum(i);
  }
  m += diff;
  arma::cx_vec X = arma::fft(target_cp);
  arma::cx_vec Y = arma::fft(spectrum_cp);
  arma::cx_vec R = X % arma::conj(Y);
  arma::cx_vec Rx = R / m;
  arma::cx_vec rev = arma::ifft(Rx);
  arma::vec vals = arma::real(rev);
  int64_t maxpos = 0;
  double maxi = -1;
  if (m < shift) {
    shift = m;
  }
  for (int64_t i = 0; i < shift; i++) {
    // std::cerr << "maxi, shift -> " << i << '\n';
    if (vals(i) > maxi) {
      maxi = vals(i);
      maxpos = i;
    }
    if (vals(vals.n_elem - i - 1) > maxi) {
      maxi = vals(vals.n_elem - i - 1);
      maxpos = vals.n_elem - i - 1;
    }
  }
  if (maxi < 0.1) {
    lag = 0;
    return lag;
  }
  if (static_cast<double>(maxpos) > (static_cast<double>(vals.n_elem) / 2.)) {
    lag = maxpos - vals.n_elem;
  } else {
    lag = maxpos;
  }

  return lag;
}

arma::vec recurAlign(arma::vec const &spectrum, arma::vec const &reference,
                     int64_t shift, int64_t lookahead) {
  if (spectrum.n_elem < 10) {
    return spectrum;
  }
  int64_t lag = FFTcorr(spectrum, reference, shift);
  if (lag == 0 && lookahead <= 0) {
    return spectrum;
  }
  if (lag == 0) {
    lookahead -= 1;
  }
  arma::vec aligned = spectrum;
  // std::cerr << "aligned -> " << aligned.n_elem <<'\n';
  if (std::abs(lag) < static_cast<int64_t>(spectrum.n_elem)) {
    aligned = move(spectrum, lag);
  }
  // std::cerr << "aligned -> " << aligned.n_elem <<'\n';
  int64_t mid = findMid(aligned);
  arma::vec firstSH = aligned.subvec(0, mid);
  arma::vec firstRH = reference.subvec(0, mid);
  arma::vec secSH = aligned.subvec(mid + 1, aligned.n_elem - 1);
  arma::vec secRH = reference.subvec(mid + 1, reference.n_elem - 1);
  arma::vec aligned1 = recurAlign(firstSH, firstRH, shift, lookahead);
  arma::vec aligned2 = recurAlign(secSH, secRH, shift, lookahead);
  return arma::join_cols(aligned1, aligned2);
}

int main(int argc, char *argv[]) {
  std::cerr << "Loading matrix...\n";
  auto mat = arma::mat();
  mat.load(argv[1]);

  int64_t shift = 0;
  int64_t lookahead = 1;

  if (shift == 0) {
    shift = mat.n_cols - 1;
  }

  arma::vec rowsds(mat.n_rows);
  for (uint64_t i = 0; i < mat.n_rows; i++) {
    arma::vec r = mat.row(i).t();
    double sd = arma::stddev(r);
    rowsds(i) = sd;
    mat.row(i) /= sd;
  }

  std::cerr << "Finding best reference...\n";
  arma::mat cors = arma::cor(mat.t());
  arma::vec cormeans = arma::vec(cors.n_cols);
  for (uint64_t i = 0; i < cors.n_cols; i++) {
    cormeans(i) = arma::mean(cors.col(i));
  }
  uint64_t idx = cormeans.index_max();

  std::cerr << "Reference spectrum (0-indexed): " << idx << '\n';

  arma::ivec constraints = {32, 64, 128, 256, 512, 1024, 2048, 4096, mat.n_cols};

  arma::vec ref = mat.row(idx).t();
  for (uint64_t i = 0; i < mat.n_rows; i++) {
    std::cerr << "Processing sample " << i << '\n';
    arma::vec tar = mat.row(i).t();

    double best_cor = -arma::datum::inf;
    arma::vec ret;
    int64_t best_constraint = 0;

    for (auto const &constraint : constraints) {
      std::cerr << "Trying constraint " << constraint << '\n';
      arma::vec cur_ret = recurAlign(tar, ref, constraint, lookahead);
      double cur_cor = arma::as_scalar(arma::cor(cur_ret, ref));
      std::cerr << "Got cor: " << cur_cor << '\n';
      if (cur_cor > best_cor) {
        best_cor = cur_cor;
        ret = cur_ret;
        best_constraint = constraint;
        std::cerr << "Best cor: " << best_cor
                  << ", constrained at: " << constraint << '\n';
      }
    }

    ret *= rowsds(i);
    for (uint64_t j = 0; j < ret.n_elem; j++) {
      std::cout << ret(j);
      if (j == (ret.n_elem - 1)) {
        std::cout << '\n';
      } else {
        std ::cout << '\t';
      }
    }
  }

  std::cerr << "Done\n";

  return 0;
}
