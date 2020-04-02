% spectrum_loss: computes the negative log likelihood for centered
% flux y:
%
%     -log p(y | Lyα z, σ², M, ω², c₀, τₒ, β)
%   = -log N(y; 0, MM' + diag(σ² + (ω ∘ (c₀ + a(1 + Lyα z)))²)),
%
% where a(Lyα z) is the approximate absorption due to Lyman α at
% redshift z:
%
%   a(z) = 1 - exp(-τ₀(1 + z)ᵝ)
%
% and its derivatives wrt M, log ω, log c₉, log τ₀, and log β

function [nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta] = ...
      spectrum_loss_lyseries(y, lya_1pz, noise_variance, M, omega2, c_0, tau_0, beta, ...
          num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, zqso_1pz)

  log_2pi = 1.83787706640934534;

  [n, k] = size(M);

  % compute approximate Lyα optical depth/absorption
  lya_optical_depth = tau_0 .* lya_1pz.^beta;

  % compute approximate Lyman series optical depth/absorption
  % using the scaling relationship

  for i = 2:num_forest_lines
    lyman_1pz = all_transition_wavelengths(1) .* lya_1pz ...
        ./ all_transition_wavelengths(i);
    
    indicator = lyman_1pz <= zqso_1pz;
    lyman_1pz = lyman_1pz .* indicator;
    
    tau = tau_0 * all_transition_wavelengths(i) * all_oscillator_strengths(i) ...
      / ( all_transition_wavelengths(1) * all_oscillator_strengths(1) );

    lya_optical_depth = lya_optical_depth + tau .* lyman_1pz.^beta;
  end
  lya_absorption = exp(-lya_optical_depth);

  % compute "absorption noise" contribution
  scaling_factor = 1 - lya_absorption + c_0;
  absorption_noise = omega2 .* scaling_factor.^2;

  d = noise_variance + absorption_noise;

  d_inv = 1 ./ d;
  D_inv_y = d_inv .* y;
  D_inv_M = bsxfun(@times, d_inv, M);

  % use Woodbury identity, define
  %   B = (I + MᵀD⁻¹M),
  % then
  %   K⁻¹ = D⁻¹ - D⁻¹MB⁻¹MᵀD⁻¹

  B = M' * D_inv_M;
  B(1:(k + 1):end) = B(1:(k + 1):end) + 1;
  L = chol(B);
  % C = B⁻¹MᵀD⁻¹
  C = L \ (L' \ D_inv_M');

  K_inv_y = D_inv_y - D_inv_M * (C * y);

  log_det_K = sum(log(d)) + 2 * sum(log(diag(L)));

  % negative log likelihood:
  %   ½ yᵀ (K + V + A)⁻¹ y + log det (K + V + A) + n log 2π
  nlog_p = 0.5 * (y' * K_inv_y + log_det_K + n * log_2pi);

  % gradient wrt M
  K_inv_M = D_inv_M - D_inv_M * (C * M);
  dM = -(K_inv_y * (K_inv_y' * M) - K_inv_M);

  % compute diag K⁻¹ without computing full product
  diag_K_inv = d_inv - sum(C .* D_inv_M')';

  % gradient wrt log ω
  dlog_omega = -(absorption_noise .* (K_inv_y.^2 - diag_K_inv));

  % gradient wrt log c₀
  da = c_0 * omega2 .* scaling_factor;
  dlog_c_0 = -(K_inv_y .* da)' * K_inv_y + diag_K_inv' * da;

  % gradient wrt log τ₀
  da = omega2 .* scaling_factor .* lya_optical_depth .* lya_absorption;
  dlog_tau_0 = -(K_inv_y .* da)' * K_inv_y + diag_K_inv' * da;

  % gradient wrt log β
  da = da .* log(lya_1pz) * beta;
  dlog_beta  = -(K_inv_y .* da)' * K_inv_y + diag_K_inv' * da;

end
