% x is [M(:); log_omega(:)]

function [f, g] = objective(x, centered_rest_fluxes, rest_noise_variances)

  [num_quasars, num_pixels] = size(centered_rest_fluxes);

  k = numel(x) / num_pixels - 1;

  ind = (1:(num_pixels * k));
  M = reshape(x(ind), [num_pixels, k]);

  ind = (num_pixels * k + 1):(num_pixels * (k + 1));
  log_omega = x(ind);

  omega2 = exp(2 * log_omega);

  f          = 0;
  dM         = zeros(size(M));
  dlog_omega = zeros(size(log_omega));

  for i = 1:num_quasars
    ind = (~isnan(centered_rest_fluxes(i, :)));

    [this_f, this_dM, this_dlog_omega] = ...
        spectrum_loss(centered_rest_fluxes(i, ind)', M(ind, :), ...
                      omega2(ind), rest_noise_variances(i, ind)');

    f               = f               + this_f;
    dM(ind, :)      = dM(ind, :)      + this_dM;
    dlog_omega(ind) = dlog_omega(ind) + this_dlog_omega;
  end

  g = [dM(:); dlog_omega(:)];

end