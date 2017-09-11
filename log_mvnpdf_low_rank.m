% log_mvnpdf_low_rank: efficiently computes
%
%   log N(y; mu, MM' + diag(d))

function log_p = log_mvnpdf_low_rank(y, mu, M, d)

  log_2pi = 1.83787706640934534;

  [n, k] = size(M);

  y = y - mu;

  d_inv = 1 ./ d;
  D_inv_y = d_inv .* y;
  D_inv_M = d_inv .* M;

  % use Woodbury identity, define
  %   B = (I + M' D^-1 M),
  % then
  %   K^-1 = D^-1 - D^-1 M B^-1 M' D^-1

  B = M' * D_inv_M;
  B(1:(k + 1):end) = B(1:(k + 1):end) + 1;
  L = chol(B);
  % C = B^-1 M' D^-1
  C = L \ (L' \ D_inv_M');

  K_inv_y = D_inv_y - D_inv_M * (C * y);

  log_det_K = sum(log(d)) + 2 * sum(log(diag(L)));

  log_p = -0.5 * (y' * K_inv_y + log_det_K + n * log_2pi);

end