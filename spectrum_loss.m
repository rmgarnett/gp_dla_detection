% spectrum_loss: computes the negative log likelihood for centered
% flux y:
%
%     -log p(y | M, log omega, noise)
%   = -log N(y; 0, MM' + diag(omega^2 + noise^2))
%
% and its derivatives wrt M and log omega

function [nlog_p, dM, dlog_omega] = spectrum_loss(y, M, omega2, noise_variance)

  log_2pi = 1.83787706640934534;

  [n, k] = size(M);

  d = omega2 + noise_variance;

  d_inv = 1 ./ d;
  D_inv_y = d_inv .* y;
  D_inv_M = bsxfun(@times, d_inv, M);

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

  nlog_p = 0.5 * (y' * K_inv_y + log_det_K + n * log_2pi);

  % gradient wrt M
  K_inv_M = D_inv_M - D_inv_M * (C * M);
  dM = -(K_inv_y * (K_inv_y' * M) - K_inv_M);

  % gradient wrt log(omega)

  % compute diag(K^-1) without computing full product
  diag_K_inv = d_inv - sum(C .* D_inv_M')';
  dlog_omega = -(omega2 .* (K_inv_y.^2 - diag_K_inv));

end
