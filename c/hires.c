/* High-resolution sweep for Fig. 6(b,c): lean maxima and the two largest Lyapunov exponents of the weave on
   many energy levels H at constant steer torque T (benchmark).  Method:
   fixed-step RK4 on the 4-D leaf (engine mode 1), start at the given state (focus turn + roll rate 1e-2 rad/s),
   transient t_trans without tangent vectors, then t_run with m = 2 tangent vectors, QR (Gram-Schmidt) every
   renorm_t, lean maxima by cubic Hermite interpolation on the step, stop at a fall (phi > lim_up or < -0.6).
   Levels are run in parallel with OpenMP (one level per thread, dynamic schedule).
   If the first start falls and the level has a fallback state (flag 1), the level is run again from it.

   Build: gcc -O3 -march=native -fopenmp hires.c engine_model.c _generated.c -lm -o hires
   Usage: hires levels.txt out.txt [h t_trans t_run renorm_t fd]
   levels.txt: line 1 "T V0", line 2 par (26 values), line 3 mass (4 values), line 4 n,
               then n lines "idx H V lim_up y0[4] flag yfb[4]".
   out.txt: one line per level (appended, existing idx skipped):
            idx H V start status t_end lam1 lam2 b1 b2 b3 b4 nmax max_1 ... max_nmax
            status 0 bounded, 1 fall up, 2 fall down, >=10 engine failure; start 0 first, 1 fallback.
            b1..b4: lam1 over the four quarters of t_run.  maxima are lean angles (rad) over all of t_run. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
int engine(const double *y, const double *par, const double *mass, double T, double H, int mode, double *out);

typedef struct { const double *par, *mass; double T, H; } Ctx;
static int FD = 2;  /* 2: central differences (as driver.c), 1: forward differences */

static int f4(const Ctx *c, const double *y, double *f) {
  double yy[5] = {y[0], y[1], y[2], y[3], 0}, out[16];
  int code = engine(yy, c->par, c->mass, c->T, c->H, 1, out);
  if (code) return code;
  f[0] = out[0]; f[1] = out[1]; f[2] = out[2]; f[3] = out[3];
  return 0;
}
static int fext(const Ctx *c, int m, const double *x, double *dx) {
  int code = f4(c, x, dx);
  if (code) return code;
  for (int i = 0; i < m; i++) {
    const double *v = x + 4 + 4 * i;
    double n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
    if (n == 0) { memset(dx + 4 + 4 * i, 0, 4 * sizeof(double)); continue; }
    double yp[4], ym[4], fp[4], fm[4];
    if (FD == 2) {
      double e = 1e-6 / n;
      for (int k = 0; k < 4; k++) { yp[k] = x[k] + e * v[k]; ym[k] = x[k] - e * v[k]; }
      if ((code = f4(c, yp, fp))) return code;
      if ((code = f4(c, ym, fm))) return code;
      for (int k = 0; k < 4; k++) dx[4 + 4 * i + k] = (fp[k] - fm[k]) / (2 * e);
    } else {
      double e = 1e-7 / n;
      for (int k = 0; k < 4; k++) yp[k] = x[k] + e * v[k];
      if ((code = f4(c, yp, fp))) return code;
      for (int k = 0; k < 4; k++) dx[4 + 4 * i + k] = (fp[k] - dx[k]) / e;
    }
  }
  return 0;
}

/* integrate nsteps; m tangent vectors in x[4..]; S[i] += log stretch; maxima appended to mx (capacity maxev);
   S over quarters in Sq (4 x m) if Sq != NULL. returns status, *nsteps_done */
static int run(double *x, const Ctx *c, double h, long nsteps, int m, long renorm, double lim_up, double lim_down,
               double *S, double *Sq, double *mx, long maxev, long *nev, long *done) {
  int n = 4 + 4 * m, code, status = 0;
  double k1[12], k2[12], k3[12], k4[12], xt[12];
  long s, q = nsteps / 4;
  if ((code = fext(c, m, x, k1))) { *done = 0; return 10 + code; }
  for (s = 0; s < nsteps; s++) {
    double xo[4], fo[4];
    memcpy(xo, x, 4 * sizeof(double)); memcpy(fo, k1, 4 * sizeof(double));
    for (int i = 0; i < n; i++) xt[i] = x[i] + .5 * h * k1[i];
    if ((code = fext(c, m, xt, k2))) { status = 10 + code; break; }
    for (int i = 0; i < n; i++) xt[i] = x[i] + .5 * h * k2[i];
    if ((code = fext(c, m, xt, k3))) { status = 10 + code; break; }
    for (int i = 0; i < n; i++) xt[i] = x[i] + h * k3[i];
    if ((code = fext(c, m, xt, k4))) { status = 10 + code; break; }
    for (int i = 0; i < n; i++) x[i] += h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    if (m && (s + 1) % renorm == 0) {
      for (int i = 0; i < m; i++) {
        double *v = x + 4 + 4 * i;
        for (int j = 0; j < i; j++) {
          double *w = x + 4 + 4 * j, d = 0;
          for (int k = 0; k < 4; k++) d += v[k] * w[k];
          for (int k = 0; k < 4; k++) v[k] -= d * w[k];
        }
        double nn = 0;
        for (int k = 0; k < 4; k++) nn += v[k] * v[k];
        nn = sqrt(nn);
        S[i] += log(nn);
        if (Sq) { long b = s / q; if (b > 3) b = 3; Sq[b * m + i] += log(nn); }
        for (int k = 0; k < 4; k++) v[k] /= nn;
      }
    }
    if ((code = fext(c, m, x, k1))) { status = 10 + code; break; }
    if (mx && xo[2] > 0 && x[2] <= 0 && *nev < maxev) {
      double th = xo[2] / (xo[2] - x[2]);
      for (int it = 0; it < 6; it++) {
        double a = th, d00 = 6 * a * a - 6 * a, d10 = 3 * a * a - 4 * a + 1, d01 = -6 * a * a + 6 * a,
               d11 = 3 * a * a - 2 * a;
        double h00 = 2 * a * a * a - 3 * a * a + 1, h10 = a * a * a - 2 * a * a + a, h01 = -2 * a * a * a + 3 * a * a,
               h11 = a * a * a - a * a;
        double g = h00 * xo[2] + h10 * h * fo[2] + h01 * x[2] + h11 * h * k1[2];
        double dg = d00 * xo[2] + d10 * h * fo[2] + d01 * x[2] + d11 * h * k1[2];
        if (dg != 0) th -= g / dg;
        if (th < 0) th = 0;
        if (th > 1) th = 1;
      }
      double a = th, h00 = 2 * a * a * a - 3 * a * a + 1, h10 = a * a * a - 2 * a * a + a,
             h01 = -2 * a * a * a + 3 * a * a, h11 = a * a * a - a * a;
      mx[(*nev)++] = h00 * xo[0] + h10 * h * fo[0] + h01 * x[0] + h11 * h * k1[0];
    }
    if (x[0] > lim_up) { status = 1; s++; break; }
    if (x[0] < lim_down) { status = 2; s++; break; }
  }
  *done = s;
  return status;
}

typedef struct { long idx; double H, V, lim, y0[4], yfb[4]; int flag; } Level;

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: hires levels.txt out.txt [h t_trans t_run renorm_t fd]\n"); return 1; }
  double h = argc > 3 ? atof(argv[3]) : 0.004, t_trans = argc > 4 ? atof(argv[4]) : 600.,
         t_run = argc > 5 ? atof(argv[5]) : 3000., renorm_t = argc > 6 ? atof(argv[6]) : 0.5;
  FD = argc > 7 ? atoi(argv[7]) : 2;
  FILE *fi = fopen(argv[1], "r");
  if (!fi) return 2;
  double T, V0, par[26], mass[4];
  long n;
  if (fscanf(fi, "%lf %lf", &T, &V0) != 2) return 3;
  for (int i = 0; i < 26; i++) if (fscanf(fi, "%lf", par + i) != 1) return 3;
  for (int i = 0; i < 4; i++) if (fscanf(fi, "%lf", mass + i) != 1) return 3;
  if (fscanf(fi, "%ld", &n) != 1) return 3;
  Level *L = malloc(n * sizeof(Level));
  for (long i = 0; i < n; i++) {
    Level *l = L + i;
    if (fscanf(fi, "%ld %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf", &l->idx, &l->H, &l->V, &l->lim, l->y0,
               l->y0 + 1, l->y0 + 2, l->y0 + 3, &l->flag, l->yfb, l->yfb + 1, l->yfb + 2, l->yfb + 3) != 13)
      return 4;
  }
  fclose(fi);
  /* resume: skip levels already in out.txt */
  char *done_f = calloc(n, 1);
  FILE *fo = fopen(argv[2], "r");
  if (fo) {
    char *line = NULL; size_t cap = 0;
    while (getline(&line, &cap, fo) > 0) {
      long id = atol(line);
      for (long i = 0; i < n; i++) if (L[i].idx == id) done_f[i] = 1;
    }
    free(line); fclose(fo);
  }
  fo = fopen(argv[2], "a");
  long ntr = lround(t_trans / h), nrun = lround(t_run / h), renorm = lround(renorm_t / h);
  long maxev = (long)(t_run / 1.0) + 100;
  double t0 = omp_get_wtime();
  long ndone = 0;
#pragma omp parallel for schedule(dynamic, 1)
  for (long i = 0; i < n; i++) {
    if (done_f[i]) continue;
    Level *l = L + i;
    Ctx c = {par, mass, T, l->H + V0};
    double *mx = malloc(maxev * sizeof(double));
    int status = 0, start = 0;
    long nev = 0, dn = 0;
    double S[2], Sq[8], t_end = 0;
    for (start = 0; start <= l->flag; start++) {
      double x[12] = {0};
      memcpy(x, start ? l->yfb : l->y0, 4 * sizeof(double));
      S[0] = S[1] = 0; memset(Sq, 0, sizeof Sq); nev = 0;
      status = run(x, &c, h, ntr, 0, 1, l->lim, -0.6, S, NULL, NULL, 0, &nev, &dn);
      t_end = dn * h;
      if (status) continue;
      /* fixed initial tangent vectors (orthonormal, generic) */
      double Q[8] = {0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5};
      memcpy(x + 4, Q, sizeof Q);
      status = run(x, &c, h, nrun, 2, renorm, l->lim, -0.6, S, Sq, mx, maxev, &nev, &dn);
      t_end = t_trans + dn * h;
      if (!status) break;
    }
    if (start > l->flag) start = l->flag;
#pragma omp critical
    {
      fprintf(fo, "%ld %.10f %.10f %d %d %.3f", l->idx, l->H, l->V, start, status, t_end);
      if (status == 0) {
        double q = (nrun / 4) * h;
        fprintf(fo, " %.6f %.6f %.6f %.6f %.6f %.6f %ld", S[0] / t_run, S[1] / t_run, Sq[0] / q, Sq[2] / q,
                Sq[4] / q, Sq[6] / (t_run - 3 * q), nev);
        for (long k = 0; k < nev; k++) fprintf(fo, " %.9f", mx[k]);
      } else
        fprintf(fo, " nan nan nan nan nan nan 0");
      fprintf(fo, "\n");
      fflush(fo);
      ndone++;
      if (ndone % 50 == 0) {
        fprintf(stderr, "%ld levels, %.0f s\n", ndone, omp_get_wtime() - t0);
        fflush(stderr);
      }
    }
    free(mx);
  }
  fclose(fo);
  fprintf(stderr, "done %ld levels in %.0f s\n", ndone, omp_get_wtime() - t0);
  return 0;
}
