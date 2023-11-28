# Реализация второго задания по курсу "Суперкомпьютерное моделирование и технологии" 2 курса магистратуры ВМК МГУ
Отчет о выполнении задания: [HPC_Task_2.pdf](./HPC_Task_2.pdf)

Сборка и запуск на IBM Polus:
```bash
g++ -o main.out main.cpp -std=c++11 -fopenmp
mpisubmit.pl -t 8 main.out --stdout std.out --stderr std.err -- 128 20 1. 1. 1. 0.01
```
Визуализация поведения решения и ошибки для сетки $`L_x=1, L_y=1, L_z=1, T=1`$, со 128 пространственным и 256 временными шагами:

![Поведение решения](./visualisations/calculated.gif)
![Ошибка](./visualisations/error.gif)
