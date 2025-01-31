import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

N = 100
sigma_v = 10


x = np.linspace(0, 4 * np.pi, N) 
sin_signal = np.sin(x)  

initial_guess = np.zeros(N)  
A = np.zeros((N, N))

#A = np.random.randint(1, 6, size=(N, N))


#for i in range(N):
#    for j in range(max(0, i-2), min(N, i+2 + 1)):
#        A[i,j] = 0.75

for i in range(N):
    A[i, i] = 1 + np.sin(i * 0.1)/15

    if i - 1 >= 0:
        A[i - 1, i] = 7 + np.cos(i * 0.5) /15
    if i + 1 < N:
        A[i + 1, i] = 7 + np.cos(i * 0.5)/15

    if i - 2 >= 0:
        A[i - 2, i] = 5 + np.sin(i * 0.3)/15
    if i + 2 < N:
        A[i + 2, i] = 5 + np.sin(i * 0.3)/15

    if i - 3 >= 0:
        A[i - 3, i] = 3 + np.cos(i * 0.2)/15
    if i + 3 < N:
        A[i + 3, i] = 3 + np.cos(i * 0.2)/15

    if i - 4 >= 0:
        A[i - 4, i] = 0 + np.sin(i * 0.1)/15
    if i + 4 < N:
        A[i + 4, i] = 0 + np.sin(i * 0.1)/15


noise = np.random.normal(0, sigma_v, N)

F = (1**2) * np.eye(N)  
T = (2**2) * np.eye(N)  

observed_signal = A @ sin_signal + noise

U = np.eye(N)

AFS  = U @ initial_guess + U @ F @ np.conjugate(A.T) @ np.linalg.pinv(A @ F @ np.conjugate(A.T) + T) @ (observed_signal - A @ initial_guess)

R = U @ np.linalg.pinv(np.linalg.pinv(sqrtm(T)) @ A) @ np.linalg.pinv(sqrtm(T))
AS  = R @ observed_signal

plt.figure(figsize=(10, 6))
plt.plot(sin_signal[:100], label='Sine signal')
plt.plot(observed_signal[:100], label='Observed signal (A * f + noise)', alpha=0.7)
plt.plot(AS[:100], label='Estimation [A, sigma]', alpha=0.7)
plt.plot(AFS[:100], label='Estimation [A, f0, F, sigma]', alpha=0.7)
plt.title('Signals: Sine, Observed, and Estimated')
plt.grid()
plt.legend()
plt.show()

err_for_as = np.abs(sin_signal - AS)
error_for_afs = np.abs(sin_signal - AFS)

plt.figure(figsize=(10, 6))
plt.plot(err_for_as[:100], label='signal - [A, sigma]')
plt.plot(error_for_afs[:100], label='signal - [A, f0, F, sigma]', alpha=0.7)
plt.title('errors')
plt.grid()
plt.legend()
plt.show()

#task2
def Z(w):
    return U @ np.linalg.pinv((np.eye(N) + w * np.linalg.pinv(np.conjugate(A.T) @ np.linalg.pinv(T) @ A)))

def U_w(w):
    return Z(w) @ np.linalg.pinv(A) @ A
    
def G(w):
    return np.linalg.norm(U -  U_w(w))**2

def h(w):
    return np.trace(U_w(w) @ np.linalg.pinv(np.conjugate(A.T) @ np.linalg.pinv(T) @ A) @ U_w(w).T)

w_values = np.linspace(0, 1, 1000)
G_values = [G(w) for w in w_values]
h_values = [h(w) for w in w_values]

plt.figure(figsize=(10, 6))
plt.plot(h_values, G_values, label='G(w) vs h(w)')

plt.title('Dependency of G(w) on h(w)')
plt.xlabel('h(w)')
plt.ylabel('G(w)')
plt.grid()
plt.legend()
plt.show()

selected_w_indices = [0,500,900]  
selected_ws = [w_values[i] for i in selected_w_indices]

plt.figure(figsize=(20, 12))

for i, w in enumerate(selected_ws):
    U = U_w(w)
    
    R = U @ np.linalg.pinv(np.linalg.pinv(sqrtm(T)) @ A) @ np.linalg.pinv(sqrtm(T))
    signal  = R @ observed_signal

    our_Uw_f = U @ initial_guess + U @ F @ np.conjugate(A.T) @ np.linalg.pinv(A @ F @ np.conjugate(A.T) + T) @ (observed_signal - A @ initial_guess)
    our_Uf_f  = signal
    
    h_w = h(w)
    
    plt.plot(our_Uw_f[:100], label=f'Estimation [A, sigma], h = {h_w} ', alpha=0.7)
    
    norm_Uw_f = np.linalg.norm(our_Uf_f - our_Uw_f)**2
    
    print(f"For w={w:.2f}: ||Uf - U(w)f||^2 = {norm_Uw_f:.4f}")

plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sin_signal[:100], label='Sine signal')
plt.plot(observed_signal[:100], label='Observed signal (A * f + noise)', alpha=0.7)
plt.plot(AS[:100], label='Estimation [A, sigma]', alpha=0.7)
plt.plot(our_Uw_f[:100], label='Sint [A, sigma]', alpha=0.7)
plt.title('Comparison of Signals: Sine, Observed, and Estimated')
plt.grid()
plt.legend()
plt.show()

