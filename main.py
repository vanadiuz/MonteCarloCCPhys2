import numpy as np
import math
from random import randint, uniform
import codecs
import matplotlib.pyplot as plt
import pandas as pd
from utilities import *
from paircorrelation import pairCorrelationFunction_3D
from array import array


#INTIAL PARAMETRS
pi = math.pi
Ns = [1,2,3]
density = [[0.05, 0.05, 0.05], [0.06, 0.08, 0.01], [0.08, 0.01, 0.06], [0.01, 0.06, 0.08]]
L = 32
eps = 1
grid_size = 1
amount_of_cells = int(L / grid_size)
amount_of_particles = 0
steps = 77777
particles = dict()
grid = set()
energy = 0
energy_sum = 0
energy_data = []
start_calc_RDF_steps = 15000
calc_RDF_each_steps = 1000
graph_RDF_counter = 0


def calc_amount_of_particles():
    global amount_of_particles
    amount_of_particles = [[0 for i in range(3)] for j in range(4)]
    for i in range(0, 4):
        for j in range(0, 3):
            amount_of_particles[i][j] = density[i][j] * L**3 // (pi/6*(j+1)**3)


#Symmetrize Matrix with Energy
def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

class SymNDArray(np.ndarray):
    def __setitem__(self, indexes, value):
        (i, j) = indexes
        super(SymNDArray, self).__setitem__(indexes, value)
        super(SymNDArray, self).__setitem__(indexes[::-1], value)

def symarray(input_array):
    """
    Returns a symmetrized version of the array-like input_array.
    Further assignments to the array are automatically symmetrized.
    """
    return symmetrize(np.asarray(input_array)).view(SymNDArray)



def prepare_particles(case):
    """
    First, distribute the particles randomly, then calculate the energy matrix, then calculate the total energy.
    Grid size is grid_size(1.5). So that the particles do not intersect!
    """
    global energy_sum, energy
    particles_before_cleaning = dict()
    init_position = int(sum(amount_of_particles[case]))
    for i in range(2, -1, -1):
        a = init_position - int(amount_of_particles[case][i])
        for item in range(init_position, init_position - int(amount_of_particles[case][i]) -1, -1):
            x = randint(1, amount_of_cells)
            y = randint(1, amount_of_cells)
            z = randint(1, amount_of_cells)
            while bool(is_everything_clean(x, y, z, grid, i+1)):
                y = randint(1, amount_of_cells)
                z = randint(1, amount_of_cells)
                x = randint(1, amount_of_cells)
            particles_before_cleaning[item] = dict(type=i+1, x=x*grid_size, y=y*grid_size, z=z*grid_size)
            add_to_grid(x, y, z, grid, i+1)
        init_position -= int(amount_of_particles[case][i])
    energy = symarray(np.zeros((len(particles_before_cleaning), len(particles_before_cleaning))))
    to_delete_this_particles = set()
    for i in range(len(particles_before_cleaning)):
        for j in range(len(particles_before_cleaning)):
            if j > i:
                r = calcDistance(particles_before_cleaning[i], particles_before_cleaning[j])
                if r - (particles_before_cleaning[i]['type'] + particles_before_cleaning[j]['type']) / 2 < 0:
                    to_delete_this_particles.add(int(i))
    # remove particles which are too close!
    for item in to_delete_this_particles:
        del particles_before_cleaning[item]
    counter = 0
    for item in particles_before_cleaning:
        particles[counter] = particles_before_cleaning[item]
        f.write("atom " + str(counter) + " radius " + str(particles[counter]['type']) + " type " + str(particles[counter]['type']))
        f.write("\n")
        counter += 1
    export_all_data()
    for i in range(len(particles)):
        for j in range(len(particles)):
            if j > i:
                energy[i,j] = calcLJ(particles[i], particles[j])
                energy_sum += energy[i,j]


def is_everything_clean(x, y, z, grid, sigma):
    answer = (x,y,z) in grid
    if sigma == 1:
        return answer

    if not answer:
        for i in [-1, 0, 1]:
            if answer: break
            for j in [-1, 0, 1]:
                if answer: break
                for k in [-1, 0, 1]:
                    answer = answer or (x + i, y + j, z + k) in grid
                    if answer: break
    else:
        return True
    if sigma == 2 or sigma == 3:
        return answer

def add_to_grid(x, y, z, grid, sigma):
    if sigma == 1:
        grid.add((x,y,z))
    else:
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    grid.add((x + i, y + j, z + k))

def calcLJ(p1, p2):
    """
    calc Lennard-Jones Potential
    """
    deltax = abs(p1['x'] - p2['x'])
    if deltax > L/2:
        deltax -= L
    deltay = abs(p1['y'] - p2['y'])
    if deltay > L/2:
        deltay -= L
    deltaz = abs(p1['z'] - p2['z'])
    if deltaz > L/2:
        deltaz -= L

    r = (deltax**2 + deltay**2 + deltaz**2)**(1/2)
    sigma = (p1['type'] + p2['type']) / 2

    answer = 0.0

    if r < 3*sigma:
        answer = 4*eps * ((sigma/r)**12 - (sigma/r)**6)
        # if answer > 5:
        #     answer = 5

    return answer

def calcDistance(p1, p2):
    """
    calc Lennard-Jones Potential
    """
    deltax = abs(p1['x'] - p2['x'])
    if deltax > L/2:
        deltax -= L
    deltay = abs(p1['y'] - p2['y'])
    if deltay > L/2:
        deltay -= L
    deltaz = abs(p1['z'] - p2['z'])
    if deltaz > L/2:
        deltaz -= L

    r = (deltax**2 + deltay**2 + deltaz**2)**(1/2)

    return r

def mc():
    """
    Monte Carlo
    """
    mc_steps = 0
    energy_data.append(energy_sum) #init iteration
    rdf = 0
    counter = 0
    r = 0
    while mc_steps < steps:
        particle = step1_and_step2()
        displacement = step3(particle)
        new_energy_sum, new_energies = step4(particle)
        while step5(new_energy_sum, new_energies, particle):
            displacement = step3a(particle, displacement)
            new_energy_sum, new_energies = step4(particle)
        export_data(particle)
        energy_data.append(energy_sum)
        if mc_steps > start_calc_RDF_steps:
            if mc_steps % calc_RDF_each_steps == 0:
                counter += 1
                if type(rdf) == int:
                    rdf = calc_RDF()[0]
                else:
                    [new_rdf, r] = calc_RDF()
                    rdf = np.sum([rdf, new_rdf], axis=0)
        mc_steps += 1
    rdf = np.array(list(map(lambda  x: x / counter, rdf)))
    graph_RDF(rdf, r)


def step1_and_step2():
    """
    Step 1 and step 2: randomly choise 1,2 or 3 type of particle and randomly choise concreate particle of preset type
    """
    # type_of_particle = randint(1, len(particles) - 1) #All particles move uniformly, not just types
    # sum = amount_of_particles[0][0]
    # i = 1
    # while type_of_particle - sum >= 0:
    #     sum += amount_of_particles[0][i]
    #     i += 1
    # type_of_particle = i

    # particle = randint(0, len(particles) - 1)

    # while particles[particle]['type'] != type_of_particle:
    #     particle = randint(0, len(particles) - 1)

    particle = randint(0, len(particles) - 1)

    return particle

def step3(particle):
    """
    Step 3: randomly select the offset
    """

    deltax = uniform(-0.5, 0.5)
    deltay = uniform(-0.5, 0.5)
    deltaz = uniform(-0.5, 0.5)
    l = (deltax**2 + deltay**2 + deltaz**2)**(1/2)
    x_limit = L > particles[particle]['x'] + deltax > 0
    y_limit = L > particles[particle]['y'] + deltay > 0
    z_limit = L > particles[particle]['z'] + deltaz > 0
    while l >= 0.76 or not x_limit or not y_limit or not z_limit:
        deltax = uniform(-0.5, 0.5)
        deltay = uniform(-0.5, 0.5)
        deltaz = uniform(-0.5, 0.5)
        l = (deltax**2 + deltay**2 + deltaz**2)**(1/2)
        x_limit = L > particles[particle]['x'] + deltax > 0
        y_limit = L > particles[particle]['y'] + deltay > 0
        z_limit = L > particles[particle]['z'] + deltaz > 0

    particles[particle]['x'] += deltax
    particles[particle]['y'] += deltay
    particles[particle]['z'] += deltaz

    return (deltax, deltay, deltaz)

def step3a(particle, displacement):
    """
    Step 3: randomly REselect the offset
    """
    deltax,deltay,deltaz = displacement
    particles[particle]['x'] -= deltax
    particles[particle]['y'] -= deltay
    particles[particle]['z'] -= deltaz

    deltax = uniform(-0.5, 0.5)
    deltay = uniform(-0.5, 0.5)
    deltaz = uniform(-0.5, 0.5)
    l = (deltax**2 + deltay**2 + deltaz**2)**(1/2)
    x_limit = L > particles[particle]['x'] + deltax > 0
    y_limit = L > particles[particle]['y'] + deltay > 0
    z_limit = L > particles[particle]['z'] + deltaz > 0
    while l >= 0.76 or not x_limit or not y_limit or not z_limit:
        deltax = uniform(-0.5, 0.5)
        deltay = uniform(-0.5, 0.5)
        deltaz = uniform(-0.5, 0.5)
        l = (deltax**2 + deltay**2 + deltaz**2)**(1/2)
        x_limit = L > particles[particle]['x'] + deltax > 0
        y_limit = L > particles[particle]['y'] + deltay > 0
        z_limit = L > particles[particle]['z'] + deltaz > 0

    particles[particle]['x'] += deltax
    particles[particle]['y'] += deltay
    particles[particle]['z'] += deltaz

    return (deltax, deltay, deltaz)

def step4(particle):
    """
    Step 4: recalculate energy
    """

    new_energy_sum = energy_sum
    new_energies = []

    for i in range(len(particles)):
        if i != particle:
            new_energy_sum -= energy[i,particle]
            new_energy = calcLJ(particles[i], particles[particle])
            new_energies.append(new_energy)
            new_energy_sum += new_energy
        else:
            new_energies.append(0)

    return new_energy_sum, new_energies

def step5(new_energy_sum, new_energies, particle):
    """
    Step 5: accept or discard the displacement?
    """
    global energy_sum
    deltaE = new_energy_sum - energy_sum


    p = uniform(0, 1)

    # if deltaE < -100:
    #     energy_sum = new_energy_sum
    #     for i in range(len(particles)):
    #         if i != particle:
    #             energy[i,particle] = new_energies[i]
    #     return False

    if p < math.exp(-deltaE):
        energy_sum = new_energy_sum
        for i in range(len(particles)):
            energy[i,particle] = new_energies[i]

        return False
    else:
        return True

def export_all_data():
    f.write('timestep indexed\n')
    for particle in particles:
        f.write(str(particle) + ' ' + str(particles[particle]['x']) + ' ' + str(particles[particle]['y']) + ' ' + str(particles[particle]['z']))
        f.write("\n")

def export_data(particle):
    f.write('timestep indexed\n')
    f.write(str(particle) + ' ' + str(particles[particle]['x']) + ' ' + str(particles[particle]['y']) + ' ' + str(particles[particle]['z']))
    f.write("\n")


def graph_energy():

    plt.figure(0)
    label = ['0.05, 0.05, 0.05', '0.06, 0.08, 0.01', '0.08, 0.01, 0.06', '0.01, 0.06, 0.08']
    for i in range(4):
        df = pd.DataFrame({'x': range(steps + 1), 'y': final_energy[i] })
        plt.plot( 'x', 'y', data=df, label=label[i])
        plt.xlabel('steps')
        plt.ylabel('energy')
        plt.title('Energy')
        plt.legend(loc='upper right', numpoints=4, ncol=1, fontsize=14)

    # plt.show()


def calc_RDF():
    dr = 0.33
    rMax = 5

    plt.figure(1)
    x = []
    y = []
    z = []
    for i in particles:
        x.append(particles[i]['x'])
        y.append(particles[i]['y'])
        z.append(particles[i]['z'])

    # Compute pair correlation
    g_r, r, reference_indices = pairCorrelationFunction_3D(np.asarray(x), np.asarray(y), np.asarray(z), L, rMax, dr)

    return [g_r, r]


def graph_RDF(g_r, r):
    global graph_RDF_counter
    dr = 0.1
    rMax = L / 4
    label = ['0.05, 0.05, 0.05', '0.06, 0.08, 0.01', '0.08, 0.01, 0.06', '0.01, 0.06, 0.08']

    # Visualize
    plt.figure(1)
    plt.plot(r, g_r, label=label[graph_RDF_counter])
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.xlim( (0, rMax) )
    plt.ylim( (0, 1.05 * g_r.max()))
    plt.title('RDF')
    plt.legend(loc='upper right', numpoints=4, ncol=1, fontsize=14)

    graph_RDF_counter += 1

###### Simulation GOING HERE!!!
calc_amount_of_particles()
final_energy = []
final_particles = []

### STEPS 1-4
for i in range(4):
    #unset all data
    particles = dict()
    grid = set()
    energy = 0
    energy_sum = 0
    energy_data = []

    f = open("data_" + str(i) + ".vtf", "w", encoding="utf8")
    f.write("# Structure block\n")

    prepare_particles(i)
    mc()

    final_energy.append(energy_data)
    final_particles.append(particles)
    f.close() 

graph_energy()
plt.show()

print(amount_of_particles[0])
print(amount_of_particles[1])
print(amount_of_particles[2])
print('amount_of_cells = ', L**3 / grid_size**3)
print('E0 =', 4*5 * ((1/1.2)**12 - (1/1.2)**6))
print('E1 =',4*5 * ((1/1.95)**12 - (1/1.95)**6))
print('E1 - E0 = ', 4*5 * ((1/1.95)**12 - (1/1.95)**6) - 4*5 * ((1/1.2)**12 - (1/1.2)**6))
print('math.exp(100) = ', math.exp(100))
print('math.exp(- (E1 - E0)) = ', math.exp( - 4*5 * ((1/1.95)**12 - (1/1.95)**6) - 4*5 * ((1/1.2)**12 - (1/1.2)**6)))






