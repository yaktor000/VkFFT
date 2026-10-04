// This file is part of VkFFT and is distributed under the MIT license.
#ifndef VKFFT_UNIT_TWIDDLE_H
#define VKFFT_UNIT_TWIDDLE_H

#include <math.h>

// Independent float rounding biases some twiddle sets towards |w|^2 < 1.
// Diffuse signed squared-magnitude error across each table instead of
// minimizing individual pair errors. Reset the residual at each table boundary.
// Search the nearest floats and their immediate neighbours; break ties by
// distance from the original double pair.
// Only use this for unit-magnitude twiddles, never transformed kernel data.
typedef struct VkFFTUnitTwiddleState {
	double residual;
} VkFFTUnitTwiddleState;

static inline void vkfft_unit_twiddle_f32(VkFFTUnitTwiddleState* state, double cd, double sd, float* twiddle) {
	const float c0 = (float)cd;
	const float s0 = (float)sd;
	twiddle[0] = c0;
	twiddle[1] = s0;
	const float c[3] = { c0, nextafterf(c0, INFINITY), nextafterf(c0, -INFINITY) };
	const float s[3] = { s0, nextafterf(s0, INFINITY), nextafterf(s0, -INFINITY) };
	double bestNormError = INFINITY;
	double bestDistance = INFINITY;
	double bestSignedError = 0.0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			const double ci = (double)c[i];
			const double sj = (double)s[j];
			const double signedError = ci * ci + sj * sj - 1.0;
			const double normError = fabs(state->residual + signedError);
			const double distance = fabs(ci - cd) + fabs(sj - sd);
			if ((normError < bestNormError) ||
				((normError == bestNormError) && (distance < bestDistance))) {
				bestNormError = normError;
				bestDistance = distance;
				bestSignedError = signedError;
				twiddle[0] = c[i];
				twiddle[1] = s[j];
			}
		}
	}
	state->residual += bestSignedError;
}

#endif
