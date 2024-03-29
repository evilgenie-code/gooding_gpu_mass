# Программа для Параллельного решения задачи Ламберта
## Авторы
- Иванюхин А.В.
- Кравченко В.С.
## Введение
В работе рассматривается возможность массивного параллельного выполнения задачи Ламберта на графических процессорах. Для этой задачи использовались несколько наиболее популярных алгоритмов решения. Программная реализация рассматриваемых алгоритмов была разработана с использованием технологии CUDA, позволяющей выполнять вычисления на графическом процессоре. Особенностью данной работы является попытка использования параллельного программирования внутри итерационной схемы. Разработанные программы были протестированы на ряде типичных задач: построения изолиний прямого перелёта Земля – Марс и решение задачи перелёта к группе астероидов при заданных дате старта и длительность перелёта. Для этих задач приведены оценки времени выполнения и эффективности распараллеливания.  

## Статья по данной теме


[Массивно параллельно решение задачи Ламберта](https://iopscience.iop.org/article/10.1088/1742-6596/1925/1/012078)


## Смежные репозитории, решение задачи Ламберта с помощью GPU

- [Решение методом Суханова](https://github.com/evilgenie-code/suhanov_gpu_mass)
- [Решение методом Иццо](https://github.com/evilgenie-code/izzo-dynamically-parallel)

## Иходные алгоритмы (написанные авторами на C++)

- [Решение методом Суханова](https://github.com/evilgenie-code/suhanov_cpu)
- [Решение методом Иццо](https://github.com/evilgenie-code/izzo_cpu)
- [Решение методом Гудинга](https://github.com/evilgenie-code/gooding_cpu)
