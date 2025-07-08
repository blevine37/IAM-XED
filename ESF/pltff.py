import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fmin
import subprocess

def read_amps(file_path):
    data = []

    with open(file_path, 'r') as file:
        for line in file:
            if '#' not in line: 
                elements = [float(element)  for i, element in enumerate(line.split()) if i in {0, 2, 3}] 
                data.append(elements)

    array = np.array(data)
    return array

file_path = 'scatamp.dat'
scat_amps = read_amps(file_path)


plt.figure(0)
plt.plot(scat_amps[:, 0],(scat_amps[:, 1]**2+scat_amps[:, 2]**2))
plt.yscale('log') 

#Convert x
#cm to Bh 
scat_amps[:, 1] = scat_amps[:, 1] * 188972598.85789
scat_amps[:, 2] = scat_amps[:, 2] * 188972598.85789
# s=2ksin(theta/2)

elekin=float(subprocess.run("grep EV n.in | awk {'print $2'}", shell=True, capture_output=True, text=True).stdout)
#elekin=2e6
#elekin=2000
#elekin=3.7*1E6 #eV
elekinha=elekin/27.21 #Ha
elv=np.sqrt(2*elekinha) #E=1/2 m v**2    m_e=1  classical
print('Clas elv:',elv)
elp=elv  #p=m*v
k=elp#/(2*np.pi)  #p=hbar*k

#relativistic calculation
#we can use two ways gamma = Etotal/Erest = (mc**2+Ekin)mc**2
# or Ek=(gamma-1)*mc**2

gamma = elekinha/(1*137**2)+1
etot = 137**2+elekinha 
elp= np.sqrt((etot**2 - 137**4)/(137**2))
k=elp
elv=elp
print('Rel elv:',elv)

#last this is ELSEPA
k = np.sqrt(elekinha * (elekinha + 2*137**2))/137
elp=k
elv=elp
print('Elsepa elv:',elv)

#fit spline
#fre = interp1d(scat_amps[:, 0], scat_amps[:, 1], kind='cubic', bounds_error=False)
#fim = interp1d(scat_amps[:, 0], scat_amps[:, 2], kind='cubic', bounds_error=False)
#f = interp1d(scat_amps[:, 0], np.sqrt(scat_amps[:, 1]**2+scat_amps[:, 2]**2), kind='cubic', bounds_error=False)
fre = interp1d(scat_amps[:, 0], scat_amps[:, 1], kind='linear', bounds_error=False)
fim = interp1d(scat_amps[:, 0], scat_amps[:, 2], kind='linear', bounds_error=False)
f = interp1d(scat_amps[:, 0], np.sqrt(scat_amps[:, 1]**2+scat_amps[:, 2]**2), kind='linear', bounds_error=False)

# s = 2*sin(theta/2) / lambda  = 2*k*sin(theta/2) becasue k=1/lambda ###=  p / pi  * sin(theta/2)       since lambda = h/p = 2 pi hbar / p = 2 pi /  p
# s = 2 * k * sin(theta/2)   
sfit = np.linspace(0,25,num=1000) #in bohr
#sfit = np.linspace(0,10,num=1000) #in bohr
print(sfit[0],sfit[-1])
thetafit = 2*np.arcsin(sfit/(2*k))*180/np.pi
print('Theta:',thetafit[0],thetafit[-1])
yre = fre(thetafit)
yim = fim(thetafit)
ytot = f(thetafit)#np.sqrt(yre**2+yim**2)

#scat_amps[:, 0] = 2*k*np.sin(scat_amps[:, 0]/2*np.pi/180) 


# Plotting
#plt.scatter(scat_amps[:, 0], scat_amps[:, 1], label='1st and 3rd Columns')

plt.figure(1)
plt.plot(sfit,yre,label='Re')
plt.plot(sfit,yim,label='Im')
plt.plot(sfit,ytot,label='Tot')

plt.xlabel('q (Bh-1)')
plt.ylabel('F(q) (Bh-1)')
plt.legend()
#plt.xlim([0,10])
#plt.show()
print('Int:', np.trapz(ytot,x=sfit))

#Ang
plt.figure(2)
plt.plot(sfit/0.529177,ytot,label='Tot') #1/bh-1 to 1/ang-1

plt.xlabel('q (Ang-1)')
plt.ylabel('F(q)')
plt.legend()
#plt.xlim([0,10])

plt.show()


