// Build/run: c++ -std=c++17 synesthesia_tests/twiddle_bias.cpp -o /tmp/twiddle_bias && /tmp/twiddle_bias
#include <cmath>
#include <cstdio>

#include "../vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_UnitTwiddle.h"

int main() {
    const double pi = 3.14159265358979323846264338327950288419716939937510;
    double oldTotal = 0.0, newTotal = 0.0;
    double oldMax = 0.0, newMax = 0.0;
    int count = 0;
    std::puts("n      count       old mean        old max        new mean        new max");
    for (int n = 2; n <= 1024; n *= 2) {
        double oldSum = 0.0, newSum = 0.0;
        double oldStageMax = 0.0, newStageMax = 0.0;
        // The radix-2 stage uses exp(i * pi * j / (n / 2)), 0 <= j < n / 2.
        for (int j = 0; j < n / 2; j++) {
            const double angle = j * pi / (n / 2);
            const double cd = std::cos(angle), sd = std::sin(angle);
            const float c = static_cast<float>(cd), s = static_cast<float>(sd);
            float corrected[2];
            vkfft_unit_twiddle_f32(cd, sd, corrected);
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
        oldTotal += oldSum;
        newTotal += newSum;
        oldMax = std::fmax(oldMax, oldStageMax);
        newMax = std::fmax(newMax, newStageMax);
        count += n / 2;
    }
    std::printf(" all   %5d   % .9e   %.9e   % .9e   %.9e\n",
                count, oldTotal / count, oldMax, newTotal / count, newMax);
    // Report the requested bound honestly: local magnitude minimization does
    // not guarantee cancellation of the signed errors across a whole table.
    const bool passes = std::fabs(newTotal / count) < 1e-10;
    std::printf("Requested |new mean| < 1e-10: %s\n", passes ? "PASS" : "FAIL");
    return passes ? 0 : 1;
}
