// Build/run: c++ -std=c++17 synesthesia_tests/twiddle_bias.cpp -o /tmp/twiddle_bias && /tmp/twiddle_bias
#include <cmath>
#include <cstdio>

#include "../vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_UnitTwiddle.h"

int main() {
    const double pi = 3.14159265358979323846264338327950288419716939937510;
    double oldTotal = 0.0, newTotal = 0.0;
    double oldMax = 0.0, newMax = 0.0;
    int count = 0;
    bool tablesPass = true;
    bool neighboursPass = true;
    std::puts("n      count       old mean        old max        diffused mean   diffused max");
    for (int n = 2; n <= 1024; n *= 2) {
        VkFFTUnitTwiddleState state = { 0.0 }; // One table per n.
        double oldSum = 0.0, newSum = 0.0;
        double oldStageMax = 0.0, newStageMax = 0.0;
        // The radix-2 stage uses exp(i * pi * j / (n / 2)), 0 <= j < n / 2.
        for (int j = 0; j < n / 2; j++) {
            const double angle = j * pi / (n / 2);
            const double cd = std::cos(angle), sd = std::sin(angle);
            const float c = static_cast<float>(cd), s = static_cast<float>(sd);
            float corrected[2];
            vkfft_unit_twiddle_f32(&state, cd, sd, corrected);
            // Every component must be the nearest float or one immediate neighbour.
            for (int component = 0; component < 2; ++component) {
                const float nearest = component == 0 ? c : s;
                neighboursPass &= corrected[component] == nearest
                    || corrected[component] == std::nextafter(nearest, INFINITY)
                    || corrected[component] == std::nextafter(nearest, -INFINITY);
            }
            const double oldError = double(c) * c + double(s) * s - 1.0;
            const double newError = double(corrected[0]) * corrected[0]
                                  + double(corrected[1]) * corrected[1] - 1.0;
            oldSum += oldError;
            newSum += newError;
            oldStageMax = std::fmax(oldStageMax, std::fabs(oldError));
            newStageMax = std::fmax(newStageMax, std::fabs(newError));
        }
        std::printf("%4d   %5d   % .9e   %.9e   % .9e   %.9e\n",
                    n, n / 2, oldSum / (n / 2), oldStageMax,
                    newSum / (n / 2), newStageMax);
        if (std::fabs(newSum / (n / 2)) >= 1e-10) {
            if (n >= 16) tablesPass = false;
            std::printf("  n=%d misses |diffused mean| < 1e-10\n", n);
        }
        oldTotal += oldSum;
        newTotal += newSum;
        oldMax = std::fmax(oldMax, oldStageMax);
        newMax = std::fmax(newMax, newStageMax);
        count += n / 2;
    }
    std::printf(" all   %5d   % .9e   %.9e   % .9e   %.9e\n",
                count, oldTotal / count, oldMax, newTotal / count, newMax);
    const bool overallPass = std::fabs(newTotal / count) < 1e-10;
    std::printf("Requested |diffused mean| < 1e-10 for every n >= 16: %s\n",
                tablesPass ? "PASS" : "FAIL");
    std::printf("Requested |overall diffused mean| < 1e-10: %s\n",
                overallPass ? "PASS" : "FAIL");
    std::printf("Components within one neighbour of nearest float: %s\n",
                neighboursPass ? "PASS" : "FAIL");
    return tablesPass && overallPass && neighboursPass ? 0 : 1;
}
