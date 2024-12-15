// All eigenvalues shifting (cut-off remained)
double shift_value = 1e-5;
for(int i=0; i<n; i++) {
    S_eval[i] += shift_value; 
}

// Tikhonov
double lambda = 1e-5;
for(int i=0; i<n; i++) {
    S[i*n+i] += lambda; 
}

// Scaling eigenvectors with low eigenvalues (no cut-off)
double scaling_factor = 0.1;
for(int i=0; i<n; i++) {
    if(fabs(S_eval[i]) < 1e-5) {
        for(int j=0; j<n; j++) {
            S[i*n+j] *= scaling_factor; 
        }
    }
}

// Regularization of eigenvalues
double regularization_factor = 1e-5;
for(int i=0; i<n; i++) {
    if(fabs(S_eval[i]) < 1e-5) {
        S_eval[i] += regularization_factor; 
    }
}

//FFT (cut-off remained)

// Complex FFT implementation
void fft(std::vector<std::complex<double>>& x) {
    const int N = x.size();
    if (N <= 1) return;


    std::vector<std::complex<double>> even(N/2), odd(N/2);
    for (int i = 0; i < N/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i+1];
    }


    fft(even);
    fft(odd);


    for (int k = 0; k < N/2; k++) {
        std::complex<double> t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N/2] = even[k] - t;
    }
}

// Inverse FFT
void ifft(std::vector<std::complex<double>>& x) {

    for (auto& val : x) {
        val = std::conj(val);
    }
    

    fft(x);
    

    const int N = x.size();
    for (auto& val : x) {
        val = std::conj(val) / static_cast<double>(N);
    }
}

int reconstruct_low_eigenvalues_fft(double* S, double* ev, int n, double threshold) {

    int padded_n = nextPowerOf2(n);
    

    std::vector<std::vector<std::complex<double>>> data(padded_n, 
        std::vector<std::complex<double>>(padded_n));
    

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            data[i][j] = std::complex<double>(S[i*n + j], 0.0);
        }

        for (int j = n; j < padded_n; j++) {
            data[i][j] = std::complex<double>(0.0, 0.0);
        }
    }

    for (int i = n; i < padded_n; i++) {
        for (int j = 0; j < padded_n; j++) {
            data[i][j] = std::complex<double>(0.0, 0.0);
        }
    }
    

    for (int i = 0; i < padded_n; i++) {
        fft(data[i]);
    }
    

    for (int j = 0; j < padded_n; j++) {
        std::vector<std::complex<double>> col(padded_n);
        for (int i = 0; i < padded_n; i++) {
            col[i] = data[i][j];
        }
        fft(col);
        for (int i = 0; i < padded_n; i++) {
            data[i][j] = col[i];
        }
    }
    

    int cutoff = padded_n / 8;  
    for (int i = 0; i < padded_n; i++) {
        for (int j = 0; j < padded_n; j++) {

            int di = i < padded_n/2 ? i : padded_n - i;
            int dj = j < padded_n/2 ? j : padded_n - j;
            double dist = std::sqrt(di*di + dj*dj);
            

            if (dist > cutoff) {
                double factor = std::exp(-(dist - cutoff)*(dist - cutoff)/(2*cutoff*cutoff));
                data[i][j] *= factor;
            }
        }
    }
    

    for (int j = 0; j < padded_n; j++) {
        std::vector<std::complex<double>> col(padded_n);
        for (int i = 0; i < padded_n; i++) {
            col[i] = data[i][j];
        }
        ifft(col);
        for (int i = 0; i < padded_n; i++) {
            data[i][j] = col[i];
        }
    }
    

    for (int i = 0; i < padded_n; i++) {
        ifft(data[i]);
    }
    

    for (int i = 0; i < n; i++) {
        if (std::fabs(ev[i]) < threshold) {
            for (int j = 0; j < n; j++) {

                S[i*n + j] = std::real(data[i][j]);
            }
        }
    }
    
    return 0;
}

reconstruct_low_eigenvalues_fft(S, S_eval, n, 1e-5); 