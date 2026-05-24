import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('images', exist_ok=True)

def exact_solution(t):
    return (t + 1)**2 - 0.5 * np.exp(t)

def f(t, y):
    return y - t**2 + 1

def euler_method(f, t0, y0, h, n):
    t = np.zeros(n+1)
    y = np.zeros(n+1)
    t[0] = t0
    y[0] = y0
    for i in range(n):
        t[i+1] = t[i] + h
        y[i+1] = y[i] + h * f(t[i], y[i])
    return t, y

def runge_kutta_4(f, t0, y0, h, n):
    t = np.zeros(n+1)
    y = np.zeros(n+1)
    t[0] = t0
    y[0] = y0
    for i in range(n):
        t[i+1] = t[i] + h
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

t0 = 0
y0 = 0.5
h = 0.1
n = 20

t_exact = np.linspace(0, 2, 100)
y_exact = exact_solution(t_exact)
t_euler, y_euler = euler_method(f, t0, y0, h, n)
t_rk4, y_rk4 = runge_kutta_4(f, t0, y0, h, n)

# График 1: сравнение методов
plt.figure(figsize=(10, 6))
plt.plot(t_exact, y_exact, 'k-', label='Точное решение', linewidth=2)
plt.plot(t_euler, y_euler, 'ro--', label='Метод Эйлера', linewidth=1.5, markersize=4)
plt.plot(t_rk4, y_rk4, 'bs--', label='Рунге-Кутта 4', linewidth=1.5, markersize=4)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Сравнение численных методов')
plt.legend()
plt.grid(True)
plt.savefig('images/comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# График 2: ошибка
y_exact_at_points = exact_solution(t_euler)
error_euler = np.abs(y_euler - y_exact_at_points)
error_rk4 = np.abs(y_rk4 - y_exact_at_points)

plt.figure(figsize=(10, 6))
plt.semilogy(t_euler, error_euler, 'ro-', label='Ошибка метода Эйлера', linewidth=1.5, markersize=4)
plt.semilogy(t_rk4, error_rk4, 'bs-', label='Ошибка метода Рунге-Кутты 4', linewidth=1.5, markersize=4)
plt.xlabel('t')
plt.ylabel('Ошибка')
plt.title('Сравнение ошибок методов')
plt.legend()
plt.grid(True)
plt.savefig('images/error.png', dpi=150, bbox_inches='tight')
plt.close()

print("Готово! images/comparison.png и images/error.png созданы.")
