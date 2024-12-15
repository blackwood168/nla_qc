# ✨ Сшивая гиперповерхности 🧮 ✨
## 🔍 Описание проблемы 

В области вычислительной химии 🧪 часто возникает важная задача поиска собственных значений эрмитовых матриц. При этом в некоторых областях пространства волновых функций 〽️ появляются околонулевые собственные значения, которые приводят к неустойчивости метода расчета и мешают корректно находить минимум энергии системы ⚡️.

Квантово-химические расчеты в данной работе проводились для молекулы :

<div style="display: flex; justify-content: center;">
    <img src="plots/ml1.png" width="400" alt="Molecule structure 1"/>
    <img src="plots/ml2.png" width="400" alt="Molecule structure 2"/>
</div>


## 🛠️ Реализованные методы 

В рамках проекта были реализованы следующие методы обработки матриц и собственных значений ⚙️:

- Сдвиг всех собственных значений 
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

- Регуляризация Тихонова 
```
double lambda = 1e-5;
for(int i=0; i<n; i++) {
    S[i*n+i] += lambda; 
}
```

- Умножение диагональных элементов на регуляризационный коэффициент
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

- FFT-фильтрация 

## 📊 Результаты 

Ниже представлены результаты применения различных методов:

![](plots/our_methods.png)
*Убывание градиентов и изменений энергии*

![](plots/selected_methods.png) 
*Убывание градиентов лучших методов*


Как видно из графиков, применение методов регуляризации повышает устойчивость вычислений.


## 📚 Список литературы 

- Glebov IO, Poddubnyy VV, Khokhlov D. Perturbation theory in the complete degenerate active space (CDAS-PT2). J Chem Phys. 2024. 161(2)
- Lloyd N. Trefethen, Bau David. Numerical Linear Algebra
- Gene H. Holub, Charles F. Van Loan. Matrix Computations
