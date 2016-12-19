/* voigt.c: computes exact Voigt profiles in terms of the complex
   error function (requires libcerf) */

#include "mex.h"
#include <cerf.h>
#include <math.h>

#define LAMBDAS_ARG    prhs[0]
#define Z_ARG          prhs[1]
#define N_ARG          prhs[2]
#define NUM_LINES_ARG  prhs[3]

#define PROFILE_ARG    plhs[0]

/* number of lines in the Lyman series to consider */
#define NUM_LINES 31

/* note: all units are CGS */

/* physical constants */

const double c   = 2.99792458e+10;               /* speed of light          cm s^-1       */
const double k   = 1.38064852e-16;               /* boltzmann constant      erg K^-1      */
const double m_p = 1.672621898e-24;              /* proton mass             g             */
const double m_e = 9.10938356e-28;               /* electron mass           g             */
/* e = 1.6021766208e-19 * c / 10; */
const double e   = 4.803204672997660e-10;        /* elementary charge       statC         */

/* Lyman series */

const double transition_wavelengths[] =          /* transition wavelengths  cm            */
  {
    1.2156701e-05,
    1.0257223e-05,
    9.725368e-06,
    9.497431e-06,
    9.378035e-06,
    9.307483e-06,
    9.262257e-06,
    9.231504e-06,
    9.209631e-06,
    9.193514e-06,
    9.181294e-06,
    9.171806e-06,
    9.16429e-06,
    9.15824e-06,
    9.15329e-06,
    9.14919e-06,
    9.14576e-06,
    9.14286e-06,
    9.14039e-06,
    9.13826e-06,
    9.13641e-06,
    9.13480e-06,
    9.13339e-06,
    9.13215e-06,
    9.13104e-06,
    9.13006e-06,
    9.12918e-06,
    9.12839e-06,
    9.12768e-06,
    9.12703e-06,
    9.12645e-06
  };

const double oscillator_strengths[] =            /* oscillator strengths    dimensionless */
  {
    0.416400,
    0.079120,
    0.029000,
    0.013940,
    0.007799,
    0.004814,
    0.003183,
    0.002216,
    0.001605,
    0.00120,
    0.000921,
    0.0007226,
    0.000577,
    0.000469,
    0.000386,
    0.000321,
    0.000270,
    0.000230,
    0.000197,
    0.000170,
    0.000148,
    0.000129,
    0.000114,
    0.000101,
    0.000089,
    0.000080,
    0.000071,
    0.000064,
    0.000058,
    0.000053,
    0.000048
  };

const double Gammas[] =                          /* transition rates        s^-1          */
  {
    6.265e+08,
    1.897e+08,
    8.127e+07,
    4.204e+07,
    2.450e+07,
    1.236e+07,
    8.255e+06,
    5.785e+06,
    4.210e+06,
    3.160e+06,
    2.432e+06,
    1.911e+06,
    1.529e+06,
    1.243e+06,
    1.024e+06,
    8.533e+05,
    7.186e+05,
    6.109e+05,
    5.237e+05,
    4.523e+05,
    3.933e+05,
    3.443e+05,
    3.030e+05,
    2.679e+05,
    2.382e+05,
    2.127e+05,
    1.907e+05,
    1.716e+05,
    1.550e+05,
    1.405e+05,
    1.277e+05
  };

/* assumed constant */

const double T = 1e+04;                          /* gas temperature         K             */

/* derived constants */

/* b = sqrt(2 * k * T / m_p); */
const double b     = 1.28486551932562422e+06;    /* Doppler parameter       cm s^-1       */
/* sigma = b / M_SQRT2; */
const double sigma = 9.08537121627923800e+05;    /* Gaussian width          cm s^-1       */

/* leading_constants[i] =
       M_PI * e * e * oscillator_strengths[i] * transition_wavelengths[i] / (m_e * c);
*/
const double leading_constants[] =               /* leading constants       cm^2          */
  {
    1.34347262962625339e-07,
    2.15386482180851912e-08,
    7.48525170087141461e-09,
    3.51375347286007472e-09,
    1.94112336271172934e-09,
    1.18916112899713152e-09,
    7.82448627128742997e-10,
    5.42930932279390593e-10,
    3.92301197282493829e-10,
    2.92796010451409027e-10,
    2.24422239410389782e-10,
    1.75895684469038289e-10,
    1.40338556137474778e-10,
    1.13995374637743197e-10,
    9.37706429662300083e-11,
    7.79453203101192392e-11,
    6.55369055970184901e-11,
    5.58100321584169051e-11,
    4.77895916635794548e-11,
    4.12301389852588843e-11,
    3.58872072638707592e-11,
    3.12745536798214080e-11,
    2.76337116167110415e-11,
    2.44791750078032772e-11,
    2.15681362798480253e-11,
    1.93850080479346101e-11,
    1.72025364178111889e-11,
    1.55051698336865945e-11,
    1.40504672409331934e-11,
    1.28383057589411395e-11,
    1.16264059622218997e-11
  };

/* gammas[i] = Gammas[i] * transition_wavelengths[i] / (4 * M_PI); */
const double gammas[] =                          /* Lorentzian widths       cm s^-1       */
  {
    6.06075804241938613e+02,
    1.54841462408931704e+02,
    6.28964942715328164e+01,
    3.17730561586147395e+01,
    1.82838676775503330e+01,
    9.15463131005758157e+00,
    6.08448802613156925e+00,
    4.24977523573725779e+00,
    3.08542121666345803e+00,
    2.31184525202557767e+00,
    1.77687796208123139e+00,
    1.39477990932179852e+00,
    1.11505539984541979e+00,
    9.05885451682623022e-01,
    7.45877170715450677e-01,
    6.21261624902197052e-01,
    5.22994533400935269e-01,
    4.44469874827484512e-01,
    3.80923210837841919e-01,
    3.28912390446060132e-01,
    2.85949711597237033e-01,
    2.50280032040928802e-01,
    2.20224061101442048e-01,
    1.94686521675913549e-01,
    1.73082093051965591e-01,
    1.54536566013816490e-01,
    1.38539175663870029e-01,
    1.24652675945279762e-01,
    1.12585442799479921e-01,
    1.02045988802423507e-01,
    9.27433783998286437e-02
  };

void mexFunction(int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {

  double *lambdas, *profile, *multipliers;
  double z, N, velocity, total;
  int num_lines, i, j;
  mwSize num_points;

  /* get input */
  lambdas = mxGetPr(LAMBDAS_ARG);                /* wavelengths             A             */
  z       = mxGetScalar(Z_ARG);                  /* redshift                dimensionless */
  N       = mxGetScalar(N_ARG);                  /* column density          cm^-2         */

  num_lines = (nrhs > 3) ? (int)(mxGetScalar(NUM_LINES_ARG)) : NUM_LINES;

  /* initialize output */
  PROFILE_ARG = mxCreateDoubleMatrix(mxGetM(LAMBDAS_ARG), mxGetN(LAMBDAS_ARG), mxREAL);
  profile = mxGetPr(PROFILE_ARG);                /* absorption profile      dimensionless */

  num_points = mxGetM(LAMBDAS_ARG) * mxGetN(LAMBDAS_ARG);

  multipliers = mxMalloc(num_lines * sizeof(double));
  for (i = 0; i < num_lines; i++)
    multipliers[i] = c / (transition_wavelengths[i] * (1 + z)) / 1e8;

  for (i = 0; i < num_points; i++) {
    /* apply each absorption line */
    total = 0;
    for (j = 0; j < num_lines; j++) {
      /* velocity relative to transition wavelength */
      velocity = lambdas[i] * multipliers[j] - c;
      total += -leading_constants[j] * voigt(velocity, sigma, gammas[j]);
    }

    profile[i] = exp(N * total);
  }

  mxFree(multipliers);

}
