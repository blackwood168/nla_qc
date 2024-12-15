# Проект по обработке околонулевых собственных значений матриц 🧮
## Описание проблемы 🔍

В области вычислительной химии часто возникает задача поиска собственных значений эрмитовых матриц. При этом в некоторых областях пространства волновых функций появляются околонулевые собственные значения, которые приводят к неустойчивости метода расчета и мешают корректно находить минимум энергии системы.

## Реализованные методы 🛠️

В рамках проекта были реализованы следующие методы обработки матриц и собственных значений:

- Сдвиг всех собственных значений 📈
```
double shift_value = 1e-5;
for(int i=0; i<n; i++) {
    S_eval[i] += shift_value; 
}
```

- Сдвиг малых собственных значений
```
double regularization_factor = 1e-5;
for(int i=0; i<n; i++) {
    if(fabs(S_eval[i]) < 1e-5) {
        S_eval[i] += regularization_factor; 
    }
}
```

- Регуляризация Тихонова ⚖️
```
double lambda = 1e-5;
for(int i=0; i<n; i++) {
    S[i*n+i] += lambda; 
}
```

- Умножение диагональных элементов на регуляризационный коэффициент.
```
double scaling_factor = 0.1;
for(int i=0; i<n; i++) {
    if(fabs(S_eval[i]) < 1e-5) {
        for(int j=0; j<n; j++) {
            S[i*n+j] *= scaling_factor; 
        }
    }
}
```

- FFT-фильтрация 🌊

## Результаты 📊


## Список литературы 📚

- Glebov IO, Poddubnyy VV, Khokhlov D. Perturbation theory in the complete degenerate active space (CDAS-PT2). J Chem Phys. 2024. 161(2)
- Lloyd N. Trefethen, Bau David. Numerical Linear Algebra
- Gene H. Holub, Charles F. Van Loan. Matrix Computations
