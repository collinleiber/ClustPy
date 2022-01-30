/*   ALGORITHM AS 217 APPL. STATIST. (1985) VOL.34, NO.3

     @article{HarP85,
     author = {P. M. Hartigan},
     title = {Computation of the Dip Statistic to Test for Unimodality},
     year = 1985,
     journal = {Applied Statistics},
     pages = {320--325},
     volume = 34 }
     @article{HarJH85,
     author = {J. A. Hartigan and P. M. Hartigan},
     title = {The Dip Test of Unimodality},
     year = 1985,
     journal = {Ann. of Statistics},
     pages = {70--84},
     volume = 13 }

     Does the dip calculation for an ordered vector X using the
     greatest convex minorant and the least concave majorant, skipping
     through the data using the change points of these distributions.

     It returns the dip statistic 'DIP' and the modal interval (XL, XU).
     ===			     ======

     dip.f -- translated by f2c (version of 22 July 1992	22:54:52).

     Pretty-Edited and extended (debug argument)
     by Martin Maechler <maechler@stat.math.ethz.ch>
     ETH Seminar fuer Statistik
     8092 Zurich	 SWITZERLAND

     ---------------

     Two Bug Fixes:
     =========

     1)	July 30 1994 : For unimodal data, gave "infinite loop"  (end of code)
     2)	Oct  31 2003 : Yong Lu <lyongu+@cs.cmu.edu> : ")" typo in Fortran
     gave wrong result (larger dip than possible) in some cases
     $Id: dip.c,v 1.26 2012/08/13 16:44:11 maechler Exp $
*/

/*
Compile Windows: cc -fPIC -shared -std=c99 -o dip.dll dip.c
Compile Linux: cc -fPIC -shared -o dip.so dip.c
*/

#include <stdio.h>

/* Subroutine */
void diptst(const double x[], const int *n_,
        double *dip, int *lo_hi, int *modaltriangle,
        int *gcm, int *lcm, int *mn, int *mj, const int *debug)
{
#define low   lo_hi[0]
#define high  lo_hi[1]
#define best_low   lo_hi[2]
#define best_high  lo_hi[3]
#define modaltriangle_i1  modaltriangle[0]
#define modaltriangle_i2  modaltriangle[1]
#define modaltriangle_i3  modaltriangle[2]

    const int n = *n_;
    int mnj, mnmnj, mjk, mjmjk, ig, ih, iv, ix,  i, j, k, l_lcm, l_gcm;
    double dip_l, dip_u, dipnew;

/*-------- Function Body ------------------------------ */

/* Check that X is sorted --- if not, return with  ifault = 2*/
    for (k = 1; k <= (n - 1); ++k) if (x[k] < x[k - 1]) return;

/* Check for all values of X identical, */
/*     and for 1 <= n < 4. */

/* LOW contains the index of the current estimate  of the lower end
   of the modal interval, HIGH contains the index for the upper end.
*/
    low = 0;    high = n - 1; /*-- IDEA:  *xl = x[low];    *xu = x[high]; --*/

    best_low = -1;
    best_high = -1;
    modaltriangle_i1 = -1;
    modaltriangle_i2 = -1;
    modaltriangle_i3 = -1;

/* M.Maechler -- speedup: it saves many divisions by n when we just work with
 * (2n * dip) everywhere but the very end! */
    *dip = 0.;
    if (n < 4 || x[n - 1] == x[0])      goto L_END;
 //    if(*debug == 1)
    // printf("dip() in C: n = %d; starting with  2N*dip = %g.\n",
    //  n, *dip);

/* Establish the indices   mn[1..n]  over which combination is necessary
   for the convex MINORANT (GCM) fit.
*/
    mn[0] = 0;
    for (j = 1; j <= (n - 1); ++j) {
    mn[j] = j - 1;
    while(1) {
      mnj = mn[j];
      mnmnj = mn[mnj];
      if (mnj == 0 ||
          ( x[j]  - x[mnj]) * (mnj - mnmnj) <
          (x[mnj] - x[mnmnj]) * (j - mnj)) break;
      mn[j] = mnmnj;
    }
    }

/* Establish the indices   mj[1..n]  over which combination is necessary
   for the concave MAJORANT (LCM) fit.
*/
    mj[n - 1] = n - 1;
    for (k = n - 2; k >= 0; k--) {
    mj[k] = k + 1;
    while(1) {
      mjk = mj[k];
      mjmjk = mj[mjk];
      if (mjk == (n - 1) ||
          ( x[k]  - x[mjk]) * (mjk - mjmjk) <
          (x[mjk] - x[mjmjk]) * (k - mjk)) break;
      mj[k] = mjmjk;
    }
    }

/* ----------------------- Start the cycling. ------------------------------- */
LOOP_Start:

    /* Collect the change points for the GCM from HIGH to LOW. */
    gcm[0] = high;
    for(i = 0; gcm[i] > low; i++)
    gcm[i+1] = mn[gcm[i]];
    ig = l_gcm = i; // l_gcm == relevant_length(GCM)
    ix = ig - 1; //  ix, ig  are counters for the convex minorant.


    /* Collect the change points for the LCM from LOW to HIGH. */
    lcm[0] = low;
    for(i = 0; lcm[i] < high; i++)
    lcm[i+1] = mj[lcm[i]];
    ih = l_lcm = i; // l_lcm == relevant_length(LCM)
    iv = 1; //  iv, ih  are counters for the concave majorant.

    if(*debug == 1) {
    printf("'dip': LOOP-BEGIN: 2n*D= %-8.5g  [low,high] = [%3d,%3d]", *dip, low,high);
    printf(" :\n gcm[0:%d] = ", l_gcm);
    for(i = 0; i <= l_gcm; i++) printf("%d%s", gcm[i], (i < l_gcm)? ", " : "\n");
    printf(" lcm[0:%d] = ", l_lcm);
    for(i = 0; i <= l_lcm; i++) printf("%d%s", lcm[i], (i < l_lcm)? ", " : "\n");
    }

/*  Find the largest distance greater than 'DIP' between the GCM and
 *  the LCM from LOW to HIGH. */

    // FIXME: <Rconfig.h>  should provide LDOUBLE or something like it
    long double d = 0.;// <<-- see if this makes 32-bit/64-bit difference go..
    if (l_gcm != 1 || l_lcm != 1) {
    if(*debug == 1) printf("  while(gcm[ix] != lcm[iv]) :\n");
      do { /* gcm[ix] != lcm[iv]  (after first loop) */
      long double dx;
      int gcmix = gcm[ix],
          lcmiv = lcm[iv];
      if (gcmix > lcmiv) {
          /* If the next point of either the GCM or LCM is from the LCM,
           * calculate the distance here. */
          int gcmi1 = gcm[ix + 1];
          dx = (lcmiv - gcmi1 + 1) -
          ((long double) x[lcmiv] - x[gcmi1]) * (gcmix - gcmi1)/(x[gcmix] - x[gcmi1]);
          ++iv;
          if (dx >= d) {
          d = dx;
          ig = ix + 1;
          ih = iv - 1;
          if(*debug == 1) printf(" L(%d,%d)", ig,ih);
          }
      }
      else {
          /* If the next point of either the GCM or LCM is from the GCM,
           * calculate the distance here. */
          int lcmiv1 = lcm[iv - 1];
/* Fix by Yong Lu {symmetric to above!}; original Fortran: only ")" misplaced! :*/
          dx = ((long double)x[gcmix] - x[lcmiv1]) * (lcmiv - lcmiv1) /
          (x[lcmiv] - x[lcmiv1]) - (gcmix - lcmiv1 - 1);
          --ix;
          if (dx >= d) {
          d = dx;
          ig = ix + 1;
          ih = iv;
          if(*debug == 1) printf(" G(%d,%d)", ig,ih);
          }
      }
      if (ix < 0)   ix = 0;
      if (iv > l_lcm)   iv = l_lcm;
      if(*debug == 1) {
          printf(" --> ix = %d, iv = %d\n", ix,iv);
      }
      } while (gcm[ix] != lcm[iv]);
    }
    else { /* l_gcm or l_lcm == 2 */
    d = 0.;
    if(*debug == 1)
        printf("  ** (l_lcm,l_gcm) = (%d,%d) ==> d := %g\n", l_lcm, l_gcm, (double)d);
    }


    if (d < *dip)   goto L_END;

/*     Calculate the DIPs for the current LOW and HIGH. */
    if(*debug == 1) printf("  calculating dip ..");

    int j_best, j_l = -1, j_u = -1;
    int lcm_modalTriangle_i1 = -1;
    int lcm_modalTriangle_i3 = -1;
    int gcm_modalTriangle_i1 = -1;
    int gcm_modalTriangle_i3 = -1;

    /* The DIP for the convex minorant. */
    dip_l = 0.;
    for (j = ig; j < l_gcm; ++j) {
    double max_t = 1.;
    int j_ = -1, jb = gcm[j + 1], je = gcm[j];
    if (je - jb > 1 && x[je] != x[jb]) {
      double C = (je - jb) / (x[je] - x[jb]);
      for (int jj = jb; jj <= je; ++jj) {
        double t = (jj - jb + 1) - (x[jj] - x[jb]) * C;
        if (max_t < t) {
        max_t = t; j_ = jj;
        }
      }
    }
    if (dip_l < max_t) {
        dip_l = max_t; j_l = j_;
        gcm_modalTriangle_i1 = jb;
        gcm_modalTriangle_i3 = je;
    }
    }

    /* The DIP for the concave majorant. */
    dip_u = 0.;
    for (j = ih; j < l_lcm; ++j) {
    double max_t = 1.;
    int j_ = -1, jb = lcm[j], je = lcm[j + 1];
    if (je - jb > 1 && x[je] != x[jb]) {
      double C = (je - jb) / (x[je] - x[jb]);
      for (int jj = jb; jj <= je; ++jj) {
        double t = (x[jj] - x[jb]) * C - (jj - jb - 1);
        if (max_t < t) {
        max_t = t; j_ = jj;
        }
      }
    }
    if (dip_u < max_t) {
        dip_u = max_t; j_u = j_;
        lcm_modalTriangle_i1 = jb;
        lcm_modalTriangle_i3 = je;
    }
    }

    if(*debug == 1) printf(" (dip_l, dip_u) = (%g, %g)", dip_l, dip_u);

    /* Determine the current maximum. */
    if(dip_u > dip_l) {
    dipnew = dip_u; j_best = j_u;
    } else {
    dipnew = dip_l; j_best = j_l;
    }
    if (*dip < dipnew) {
    *dip = dipnew;
    best_low = gcm[ig];
    best_high = lcm[ih];
    if (dip_u > dip_l) {
        modaltriangle_i1 = lcm_modalTriangle_i1;
        modaltriangle_i2 = j_best;
        modaltriangle_i3 = lcm_modalTriangle_i3;
        if (*debug == 1) printf(" -> new larger dip %-8.5g (j_best = %d) gcm-centred triple (%d,%d,%d)\n",dipnew, j_best,
                                                                                                      lcm_modalTriangle_i1,
                                                                                                      j_best,
                                                                                                      lcm_modalTriangle_i3);
    } else {
        modaltriangle_i1 = gcm_modalTriangle_i1;
        modaltriangle_i2 = j_best;
        modaltriangle_i3 = gcm_modalTriangle_i3;
        if (*debug == 1) printf(" -> new larger dip %-8.5g (j_best = %d) lcm-centred triple (%d,%d,%d)\n",dipnew, j_best,
                                                                                                      gcm_modalTriangle_i1,
                                                                                                      j_best,
                                                                                                      gcm_modalTriangle_i3);
    }
    }
    else if(*debug == 1) printf("\n");

    /*--- The following if-clause is NECESSARY  (may loop infinitely otherwise)!
      --- Martin Maechler, Statistics, ETH Zurich, July 30 1994 ---------- */
    if (low == gcm[ig] && high == lcm[ih]) {
      if(*debug == 1)
    printf("No improvement in  low = %d  nor  high = %d --> END\n",
        low, high);
    } else {
    low  = gcm[ig];
    high = lcm[ih];    goto LOOP_Start; /* Recycle */
    }
/*---------------------------------------------------------------------------*/

L_END:
    /* do this in the caller :
     *   *xl = x[low];  *xu = x[high];
     * rather return the (low, high) indices -- automagically via lo_hi[]  */
    *dip /= (2*n);
    return;
} /* diptst */
#undef low
#undef high
#undef best_low
#undef best_high
#undef modaltriangle_i1
#undef modaltriangle_i2
#undef modaltriangle_i3