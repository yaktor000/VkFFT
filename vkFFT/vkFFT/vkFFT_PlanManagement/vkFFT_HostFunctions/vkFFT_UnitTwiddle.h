// This file is part of VkFFT and is distributed under the MIT license.
#ifndef VKFFT_UNIT_TWIDDLE_H
#define VKFFT_UNIT_TWIDDLE_H

#include <math.h>

// Independent float rounding biases some twiddle sets towards |w|^2 < 1.
// Search the nearest floats and their immediate neighbours for the least
// squared-magnitude error; break ties by distance from the original pair.
// Only use this for unit-magnitude twiddles, never transformed kernel data.
static inline void vkfft_unit_twiddle_f32(double cd, double sd, float* twiddle) {
	const float c0 = (float)cd;
	const float s0 = (float)sd;
	const float c[3] = { c0, nextafterf(c0, INFINITY), nextafterf(c0, -INFINITY) };
	const float s[3] = { s0, nextafterf(s0, INFINITY), nextafterf(s0, -INFINITY) };
	double bestNormError = INFINITY;
	double bestDistance = INFINITY;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			const double ci = (double)c[i];
			const double sj = (double)s[j];
			const double normError = fabs(ci * ci + sj * sj - 1.0);
			const double distance = fabs(ci - cd) + fabs(sj - sd);
			if ((normError < bestNormError) ||
				((normError == bestNormError) && (distance < bestDistance))) {
				bestNormError = normError;
				bestDistance = distance;
				twiddle[0] = c[i];
				twiddle[1] = s[j];
			}
		}
	}
}

#endif
