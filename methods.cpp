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