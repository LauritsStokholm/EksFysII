# Preamble
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy import optimize as opt
from scipy import stats 

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
def radians(theta):
    return (theta*np.pi)/180

# Snells law
def snellslaw(theta1, n1, n2):
    theta2 = np.arcsin(n1*np.sin(theta1)/n2)
    return theta2

# Brewsters Angle
# Angle at which no light is reflected
def brewsterangle(n1, n2):
    return np.arctan(n2/n1)

# Critical angle
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
def Tp1(theta1, theta2):
    return np.sin(2*theta1)*np.sin(2*theta2)/((np.sin(theta1 + theta2))**2 * (np.cos(theta1 - theta2))**2)


def Tp2(theta1, theta2, n1, n2, tp):
    return np.cos(theta2)/np.cos(theta1) * n2/n1 * tp**2

# Index for reflection perpendicular
def Rs(rs):
    return rs**2

# Index for transmission perpendicular
def Ts1(theta1, theta2):
    return np.sin(2*theta1)*np.sin(2*theta2)/(np.sin(theta1+theta2))**2

def Ts2(theta1, theta2, n1, n2, ts):
    return np.cos(theta2)/np.cos(theta1) * n2/n1 * ts**2

# Theoretical
# Defining material constants
n = np.array([1, 1.5])        # air, glass

# Defining angles
theta1 = np.arange(0, np.pi/2, 0.01) # degrees
theta2 = np.zeros(np.size(theta1))

# Defining itterable matrices 
rs_theory = np.zeros(np.size(theta1))
rp_theory = np.zeros(np.size(theta1))
Rs_theory = np.zeros(np.size(theta1))
Rp_theory = np.zeros(np.size(theta1))

# Calculating theta2 from snell
itteration = np.arange(1, np.size(theta1), 1)
for angle in itteration:
    theta2[angle] = snellslaw(theta1[angle], n[0], n[1])

# Calculating Rs and Rp for plot
for i in itteration:
    rs_theory[i] = rs(theta1[i], theta2[i])
    Rs_theory[i] = Rs(rs_theory[i])

    rp_theory[i] = rp(theta1[i], theta2[i])
    Rp_theory[i] = Rp(rp_theory[i])

#theoretical transmission coefficients (mom's spaghetti)
# T = 1 - R
Ts_theory = np.ones(np.size(Rs_theory))-Rs_theory
Tp_theory = np.ones(np.size(Rp_theory))-Rp_theory

# Meassurements (raw data)
# Importing data,  delimiter = ',',
data = np.genfromtxt('data.csv', skip_header=3, comments="#", 
        dtype="float").T
print(data)

# P-polarization

# Reflection
# Angles
theta_pr1 = data[0]

# Intensity (meassured - background)
Int_use_pr1 = data[3]

# Intensity at 90 degrees (last meassurement)
Int_90 = Int_use_pr1[-1]

# Meassurements divided by maximal angle
Rp_pr1 = Int_use_pr1/Int_90
radians_pr1 = radians(theta_pr1)

# Transmission

# Angles
theta_pt1 = np.arange(0, 85+1, 5)
radians_pt1 = radians(theta_pt1)

# Intensities (meassured)
Int_pt1 = data[5]
print(Int_pt1)
#Int_pt1 = np.array([5.5, 5.15, 5.13, 5.17, 5.15, 5.13, 5.15, 5.13, 5.13, 5.13, 5.15, 5.15, 5.15, 5.15, 5.15, 4.6, 3.0, 0.84])

# Backgroind intensity
#Background_pt1 = np.ones(np.size(Int_pt1))*0.017

# Meassurement minus intensity
Int_use_pt1 = Int_pt1-Background_pt1

# Refraction intensity at 0 degrees
Int_0 = Int_use_pt1[0]

# Meassurements divided by maximum refraction
Tp_pt1 = Int_use_pt1/Int_0




#S-polarised, reflection 

theta_sr1 = np.arange(20, 80+1, 5)
radians_sr1 = radians(theta_sr1)

Int_sr1 = np.array([1.116, 1.216, 1.305, 1.606, 1.79, 2.268, 2.739, 3.6, 4.542, 5.159, 5.158, 5.158, 5.158])
Background_sr1 = np.array([0.080, 0.068, 0.070, 0.071, 0.068, 0.068, 0.067, 0.066, 0.071, 0.070, 0.073, 0.072,0.07])

Int_use_sr1 = Int_sr1-Background_sr1
Int_90_sr1 = 20
Rs_exp_ag = Int_use_sr1/Int_90_sr1





#S-polarised, transmission

theta_st1 = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,90])
radians_st1 = radians(theta_st1)
Int_st1 = np.array([1.12, 1.247, 1.1750, 1.102, 1.008, 1.019, 1.071, 1.05, 1.006, 0.997, 0.996, 0.931, 0.923, 0.873, 0.769, 0.657, 0.524, 0.357, 0.150])
Background_st1 = np.array([0.05, 0.07, 0.08316, 0.05, 0.072, 0.0727, 0.062, 0.072, 0.062, 0.062, 0.066, 0.070, 0.070, 0.07, 0.074, 0.066, 0.07, 0.067, 0.065])

Int_use_st1 = Int_st1-Background_st1
Int_0 = Int_use_st1[0]
Ts_exp_ag = Int_use_st1/Int_0





#Glass to air 

#S-polarized reflection:

theta_sr2 = np.array([20, 25, 30, 35, 40, 45, 50, 55])
radians_sr2 = radians(theta_sr2)
Int_sr2 = np.array([0.2, 0.233, 0.316, 0.455, 1.043, 3.78, 4.06, 4.03])
Background_sr2 = np.array([ 0.078, 0.071, 0.068, 0.067, 0.066, 0.070, 0.070, 0.070])

Int_use_sr2 = Int_sr2-Background_sr2
Int_90_sr2 = Int_use_sr2[-1]
Rs_exp_ga = Int_use_sr2/Int_90_sr2

#S-polarized transmission. 

theta_st2 = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
radians_st2 = radians(theta_st2)
Int_st2 = np.array([1.789, 1.739, 1.747, 1.794, 1.753, 1.717, 1.714, 1.575, 1.235])
Background_st2 = np.array([0.078, 0.072, 0.072, 0.072, 0.073, 0.073, 0.074, 0.083, 0.070])

Int_use_st2 = Int_st2 - Background_st2
Int_0_ga = Int_use_st2[0]
Ts_exp_ga = Int_use_st2/Int_0_ga


#refraction index

theta1_ref = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
theta2_ref = np.array([0, 2, 4, 5, 7, 8, 10, 12, 14, 16, 19, 21, 24, 26, 30, 33, 37, 40])
radians1_ref = radians(theta1_ref)
radians2_ref = radians(theta2_ref)
sine1 = np.sin(radians1_ref)
sine2 = np.sin(radians2_ref)

#Theoretical glass to air transmission

n_crit = criticalangle(1.5)
theta_ga = np.arange(0, n_crit, 0.01)
theta_ga_2 = snellslaw(theta_ga, 1.5, 1)
Ts_ga_theory = Ts1(theta_ga, theta_ga_2)
Ts_ga_theory = Ts_ga_theory + np.ones(np.size(Ts_ga_theory))*0.04


def poly1(x, a, b):
    return a*x+b

popt, pcov = opt.curve_fit(poly1, sine1, sine2)
print(popt)
n_2 = 1/popt[0]

#Spredninger er lort 

sds = radians(3)

def spredning(X,Y,sds):
    diff_X = np.diff(Y)/np.diff(X)
    print(diff_X)
    sigma = np.zeros(np.size(X))
    for i in range(0, np.size(diff_X)):
        sigma[i] = np.sqrt((diff_X[i]*sds)**2)
    return sigma

yerr_pr1 = spredning(radians_pr1,Rp_pr1,sds)
yerr_sr1 = spredning(radians_sr1, Rs_exp_ag, sds)
yerr_pt1 = spredning(radians_pt1, Tp_pt1, sds)
#yerr_st1 = 


# Data visualization
# Theoretic
plt.figure()
plt.plot(theta1, Rs_theory)
plt.plot(theta1, Rp_theory)
plt.xlabel(r'Angles $\theta \ [\text{radians}]$')
plt.ylabel('Rs')
plt.title('Theoretical plot')
plt.grid()


# Experimental
plt.figure()
#plt.plot(theta, Rs_ex)
plt.errorbar(radians_pr1, Rp_pr1, xerr=0, yerr=yerr_pr1)
plt.plot(theta1, Rs_theory)
plt.plot(theta1, Rp_theory)
plt.xlabel(r'Angles $\theta \ [\text{radians}]$')
plt.ylabel('Rs')
plt.grid()
plt.legend(loc=2)
plt.savefig('reflection.jpg')

#Transmission coefficients
plt.figure()
plt.title('Experimental plot (transmission)')
plt.plot(theta1, Ts_theory, label='Theoretical s-polarized')
plt.plot(theta1, Tp_theory, label='Theoretical p-polarized')
plt.plot(radians_pt1, Tp_pt1, 'ro', label='Data')
plt.xlabel(r'Angles$\theta \ [\text{radians}]$')
plt.ylabel('Ts/Tp')
plt.legend(loc=3)
plt.grid()
plt.savefig('transmission.jpg')

#fuck-up graf

plt.figure()
plt.plot(radians_sr1, Rs_exp_ag, 'go')
plt.plot(theta1, Rs_theory)
plt.xlabel(r'Angles$\theta \ [\text{radians}]$')
plt.ylabel('Rs')
plt.title('Experimental plot (transmission)')
plt.grid()

#Refraction index
plt.figure()
plt.title('Snells law')
plt.plot(sine1,sine2,'bo', label='Data')
plt.plot(sine1, poly1(sine1, *popt), label='Linear fit')
plt.xlabel(r'Angles$\sin(\theta_1) \ [\text{radians}]$')
plt.ylabel(r'Angles$\sin(\theta_2) \ [\text{radians}]$')
plt.legend(loc=2)
plt.grid()
plt.savefig('snell.jpg')

#Transmission glass to air 
plt.figure()
plt.plot(radians_st2, Ts_exp_ga, 'o')
plt.plot(theta_ga, Ts_ga_theory)
plt.xlabel(r'Angles$\theta \ [\text{radians}]$')
plt.ylabel('Ts')
plt.title('Experimental plot (transmission)')
plt.grid()



plt.show()
