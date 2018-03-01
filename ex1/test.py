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
    return np.sin(2*theta1)*np.sin(2*theta2)/(np.sin(theta1+theta2)**2)

def Ts2(theta1, theta2, n1, n2, ts):
    return np.cos(theta2)/np.cos(theta1) * n2/n1 * ts**2


# # # # # # # # # # # # # # # # # # # # # # #

# Theoretical

def theoretical():
    # Defining material constants
    n = np.array([1, 1.5])        # air, glass
    
    # Defining angles in radians
    theta1 = np.arange(0, np.pi/2, 0.01)
    theta2 = np.zeros(np.size(theta1))
    
    # Defining itterable matrices 
    rs_theory = np.zeros(np.size(theta1))
    rp_theory = np.zeros(np.size(theta1))
    Rs_theory = np.zeros(np.size(theta1))
    Rp_theory = np.zeros(np.size(theta1))
    
    # Calculating theta2 from Snell's law
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
    
    return(theta1, theta2, Rs_theory, Rp_theory)


# Meassurements (raw data)

# p-polarized light, reflection. 

# Importing data
data = np.genfromtxt('data.csv', skip_header=1, comments="#", usecols=(0,1,2,3,4), delimiter = ',').T

# Angles, intensity
theta_pr1 = data[0]
Int_use_pr1 = data[3]

# Intensity at 90 degrees (last meassurement)
Int_90 = Int_use_pr1[-1]

## Relative (Meassurements divided by max angle/ max intensity)
Rp_pr1 = Int_use_pr1/Int_90

def radians(theta):
    return (theta*np.pi)/180

radians_pr1 = radians(theta_pr1)

# # # # # # # # # # # # # # # # # # #    
# p-polarized, transmission
# # # # # # # # # # # # # # # # # # #    

# Angles
theta_pt1 = np.arange(0, 90, 5) # 0 to 85 in steps of 5

#Converted to radians
radians_pt1 = radians(theta_pt1)

# Intensities (meassured)
Int_pt1 = data[4]

# Backgroind intensity
Background_pt1 = np.ones(np.size(Int_pt1))

# Meassurement minus intensity
Int_use_pt1 = Int_pt1-Background_pt1

# Refraction intensity at 0 degrees
Int_0 = Int_use_pt1[0]

# Meassurements divided by maximum refraction
Tp_pt1 = Int_use_pt1/Int_0
#
##refraction index
#
#theta1_ref = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
#theta2_ref = np.array([0, 2, 4, 5, 7, 8, 10, 12, 14, 16, 19, 21, 24, 26, 30, 33, 37, 40])
#radians1_ref = radians(theta1_ref)
#radians2_ref = radians(theta2_ref)
#sine1 = np.sin(radians1_ref)
#sine2 = np.sin(radians2_ref)


#def poly1(x, a, b):
#    return a*x+b
#
#popt, pcov = opt.curve_fit(poly1, sine1, sine2)
#print(popt)
#n_2 = 1/popt[0]
#
## Error (konfidensinterval)
#sds = 0.5
#
##konf_minus = konf(theta1_pr1, Rp_pr1,sds)[0]
##konf_plus  = konf(theta1_pr1, Rp_pr1,sds)[1]
#sigma = konf(theta_pr1, Rp_pr1,sds)



# # # # # # # # # # # # # # # # # # # # # # #

# Data visualization
# Theoretic

theta1, theta2, Rs_theory, Rp_theory = theoretical()
plt.figure()
plt.title('Theoretical plot')
plt.plot(theta1, Rs_theory)
plt.plot(theta1, Rp_theory)
plt.xlabel(r'Angles $\theta \ [\text{radians}]$')
plt.ylabel('Rs')
plt.grid()
plt.savefig('theory.jpg')

# Experimental
plt.figure()
plt.title('Experimental plot')
#plt.plot(radians_pr1, Rp_pr1, 'ro', label='Data')
plt.errorbar(radians_pr1, Rp_pr1, fmt = 'ko', xerr = sds, yerr = sigma)
plt.plot(theta1, Rs_theory, label='Theoretical s-polarized')
plt.plot(theta1, Rp_theory, label='Theoretical p-polarized')
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


plt.show()
