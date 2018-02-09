# Preamble
import numpy as np
import matplotlib.pyplot as plt

# MatploLib koerer TeX
params = {'legend.fontsize'     : '20',
          'axes.labelsize'      : '20',
          'axes.titlesize'      : '20',
          'xtick.labelsize'     : '20',
          'ytick.labelsize'     : '20',
          'legend.numpoints'    : 1,
          'text.latex.preamble' : [r'\usepackage{siunitx}',
                                   r'\usepackage{amsmath}'],
          'axes.spines.right'   : False,
          'axes.spines.top'     : False,
          'figure.figsize'      : [8.5, 6.375],
          'legend.frameon'      : False
          }

plt.rcParams.update(params)
plt.rc('text',usetex =True)
plt.rc('font', **{'family' : "sans-serif"})

# Definitions (formulaes)
# Snells law
def snellslaw(theta1, n1, n2):
    theta2 = np.arcsin(n1*np.sin(theta1)/n2)
    return theta2

# Angle at which no light is reflected
def brewsterangle(n1, n2):
    return np.arctan(n2/n1)

# Angle for total internal reflection
def criticalangle(n):
    return np.arcsin(1/n)

# Reflection parallel
def rp(theta1, theta2):
    return np.tan(theta1 - theta2) / np.tan(theta1 + theta2)

# Transmitted parallel
def tp(theta1, theta2):
    return 2*np.cos(theta1)*np.sin(theta2)/(np.sin(theta1+theta2)*np.cos(theta1-theta2))

# Reflected perpendicular
def rs(theta1, theta2):
    return -np.sin(theta1-theta2) / np.sin(theta1+theta2)

# Transmitted perpendicular
def ts(theta1, theta2):
    return 2*np.cos(theta1)*np.sin(theta2)/np.sin(theta1+theta2)

# Index for reflection parallel
def Rp(rp):
    return rp**2

# Index for transmission parallel
def Tp(tp):
    return tp**2

# Index for reflection perpendicular
def Rs(rs):
    return rs**2

# Index for transmission perpendicular
def Ts(ts):
    return (np.cos(theta2)/np.cos(theta1))*(n2/n1)*ts**2

# Theoretical
# Defining material constants
n = np.array([1, 1.5])        # air, glass

# Defining angles
theta1 = np.arange(1, 90, 0.1) # degrees
theta2 = np.zeros(np.size(theta1))

# Defining itterable matrices 
rs_theory = np.zeros(np.size(theta1))
rp_theory = np.zeros(np.size(theta1))
Rs_theory = np.zeros(np.size(theta1))
Rp_theory = np.zeros(np.size(theta1))

# Calculating theta2 from snell
itteration = np.arange(0, np.size(theta1), 1)
for angle in itteration:
    theta2[angle] = snellslaw(theta1[angle], n[0], n[1])

# Calculating Rs and Rp for plot
for i in itteration:
    rs_theory[i] = rs(theta1[i], theta2[i])
    Rs_theory[i] = Rs(rs_theory[i])

    rp_theory[i] = rp(theta1[i], theta2[i])
    Rp_theory[i] = Rp(rp_theory[i])


# Meassurements (raw data)
Theta2 = np.array([19.47, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25])  # degrees
Ts = 0.942
Rs = 0.058 # 1 - Ts

if np.size(theta1) != np.size(theta2):
    print('Mangler data for theta2')

# Theoretic
plt.figure()
#plt.plot(theta1, Rp_theory)
plt.plot(theta1, Rs_theory)
plt.xlabel(r'Angles $\theta \ [^{\circ}]$')
plt.ylabel('y--values')
plt.title('Theoretical plot')
plt.grid()
plt.show()


