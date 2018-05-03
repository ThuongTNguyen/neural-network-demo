import numpy as np
import pandas as pd


def distances_from_xyz(xyz, Natom):
    """calculate distances from coordinates
    # Arguments
        Natom: number (nb) of atoms; integer
        xyz: coordinates; 2D numpy array of shape (nb_samples * nb_atoms, 3)
        
    # Returns
        distances_df: distance values; pandas dataframe of shape (nb_samples, nb_distances),
                    column names as [(0,1),(0,2),...]  
    """
    Nsamples = xyz.shape[0]//Natom
    distances = np.zeros(shape=(Nsamples, int(Natom*(Natom-1)/2)))
    count = 0
    for i in range(Natom):
        atom1_array = xyz[Natom*np.arange(Nsamples) + i]
        for j in range(i+1,Natom):
            atom2_array = xyz[Natom*np.arange(Nsamples) + j]
            distances[:,count] = np.sqrt(np.sum((atom1_array - atom2_array)**2, axis = 1))
            count += 1
    distances_df = pd.DataFrame(data= distances, columns = [(a,b) for a in range(Natom) for b in range(a+1,Natom)])
    return distances_df

####################################################
## switching function 
###################################################
def base_fswitch(ri,rf,r):
    """base switching function and its gradient w.r.t distance r
    # Arguments
        ri, rf: low and high cutoff values; float 
        r: distance; 1D numpy array or scalar
    
    # Returns
        value: base switching function values; 1D numpy array of the same length with r  
    """
    coef = np.pi/(rf-ri)
    temp = (1.0 + np.cos(coef*(r-ri)))/2.0
    value = temp*((r>=ri)&(r<rf)).astype(int) + (r<ri).astype(int)
    return value

def fswitch_2b(ri, rf, Natom, xyz, distances):
    """switching function for dimers
    # Arguments
        ri, rf: low and high cutoff values; float
        Natom: number of atoms (nb_atoms); integer
        xyz: coordinates; 2D numpy array of shape (nb_samples *Natom, 3)
        distances: distance values; pandas dataframe of shape (nb_samples, nb_distances)
    
    # Returns
        s: switching function values; 1D numpy array of length Nsamples (nb_samples)
    """
    Nsamples = distances.shape[0]
    s = base_fswitch(ri, rf, distances[(0,3)].values)
    return s 

def fswitch_3b(ri, rf, Natom, xyz, distances):
    """switching function for trimers
    # Arguments
        ri, rf: low and high cutoff values; float
        Natom: number of atoms (nb_atoms); integer
        xyz: coordinates; 2D numpy array of shape (nb_samples *Natom, 3)
        distances: distance values; pandas dataframe of shape (nb_samples, nb_distances)
    
    # Returns
        s: switching function values; 1D numpy array of length Nsamples (nb_samples)
    """
    s01 = base_fswitch(ri, rf, distances[(0,3)].values)
    s02 = base_fswitch(ri, rf, distances[(0,6)].values)
    s12 = base_fswitch(ri, rf, distances[(3,6)].values)
    s = s01*s02 + s01*s12 + s02*s12
    return s 


####################################################
def radial_filter(Rs, eta, Rij):
    """radial filter for symmetry functions
    # Arguments
        Rs, eta: radial symmetry function parameters; float
        Rij: distance values between two given atoms i and j; 
                1D numpy array of length Nsamples

    # Returns
        G_rad_ij: radial filter values; 1D numpy array of length nb_samples
    """
    G_rad_ij = np.exp(-eta * (Rij-Rs)**2) 
    return G_rad_ij 

def angular_filter(Rij, Rik, Rjk, eta, zeta, lambd): 
    """angular filter for angular symmetry functions
    # Arguments
        eta, zeta, lambd: angular symmetry function parameters
        Rij, Rik, Rjk: distances among three atoms i, j, k; 1D arrays of length nb_samples
        
    # Returns
        G_ang_ij: angular filter values; 1D numpy array of length nb_samples
        
    """
    cos_angle = (Rij**2 + Rik**2 - Rjk**2)/(2.0 * Rij * Rik)
    rad_filter = np.exp(-eta*(Rij + Rik + Rjk)**2) 
    G_ang_ijk = 2**(1.0-zeta) * (1.0 + lambd * cos_angle)**zeta * rad_filter 
    
                
    return G_ang_ijk  


def symmetry_function(distances, at_idx_map, Gparam_dict):
    """calculate symmetry functions from distances
    # Arguments
        distances: distance values; pandas dataframe of shape (nb_samples, nb_distances)
        at_idx_map: a mapping between atom types and atom indexes; dictionary
        Gparam_dict: symmetry function parameters; 
                        dictionary with 1st layer keys  = atom types,
                            2nd layer keys = symmetry function types (radial or angular)
                            values = 2D arrays of sym. function parameters of 
                            shape (nb_sym_functions, nb_filter_parameters)
                        
    # Returns        
       Gfunc_data: symmetry function values; 
                    dictionary with 1st layer keys = atom types,
                        2nd layer keys = atom indexes,
                        values = 2D arrays with shape=(nb_samples, nb_sym_functions)
    """
    Nsamples = distances.shape[0]
    Gfunc_data = {}
    for at_type in at_idx_map.keys():
        Gparam_rad = Gparam_dict[at_type]['rad']
        Gparam_ang = Gparam_dict[at_type]['ang']

        Gfunc_data[at_type] = {}

        rad_count = sum([Gparam_rad[t].shape[0] for t in Gparam_rad.keys()])
        ang_count = sum([Gparam_ang[t].shape[0] for t in Gparam_ang.keys()])

        for at1 in at_idx_map[at_type]:
            Gfunc_data[at_type][at1] = np.zeros((Nsamples, rad_count + ang_count)) 

            G_temp_count = 0

            # radial components
            for at2_type in Gparam_rad.keys():
                comp_count =  Gparam_rad[at2_type].shape[0]
                G_temp_component = np.zeros((Nsamples, comp_count))

                for count, values in enumerate(Gparam_rad[at2_type]):
                    for at2 in at_idx_map[at2_type][at_idx_map[at2_type]!=at1]:
                        dist = tuple(sorted([at1, at2]))
                        R12_array = distances[dist].values[:Nsamples]
                        rad_temp = radial_filter(values[0], values[1], R12_array)
                        G_temp_component[:,count] += rad_temp

                Gfunc_data[at_type][at1][:,G_temp_count:G_temp_count+comp_count] = G_temp_component
                G_temp_count += comp_count

            # ======================
            # angular components
            for atAatB_type in Gparam_ang.keys():
                comp_count = Gparam_ang[atAatB_type].shape[0]
                G_temp_component = np.zeros((Nsamples, comp_count))

                for count, values in enumerate(Gparam_ang[atAatB_type]):
                    atA_list = at_idx_map[atAatB_type[0]][at_idx_map[atAatB_type[0]]!=at1]
                    for atA in atA_list:
                        dist_1A = tuple(sorted([at1, atA]))
                        R1A_array = distances[dist_1A].values[:Nsamples]

                        if atAatB_type[0] == atAatB_type[1]:
                            atB_list = at_idx_map[atAatB_type[1]][(at_idx_map[atAatB_type[1]]!=at1) & (at_idx_map[atAatB_type[1]]>atA)]
                        else:
                            atB_list = at_idx_map[atAatB_type[1]][(at_idx_map[atAatB_type[1]]!=at1)]

                        for atB in atB_list:
                            dist_1B = tuple(sorted([at1, atB]))
                            dist_AB = tuple(sorted([atA, atB]))
                            R1B_array = distances[dist_1B].values[:Nsamples]
                            RAB_array = distances[dist_AB].values[:Nsamples]

                            ang_temp = angular_filter(R1A_array, R1B_array, RAB_array, values[0], values[1], values[2])

                            G_temp_component[:, count] += ang_temp 

                Gfunc_data[at_type][at1][:,G_temp_count:G_temp_count+comp_count] = G_temp_component
                G_temp_count += comp_count
    return Gfunc_data 

##################################
###### BPNN models ######
##################################
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda,floatX=float64,dnn.enabled=True"
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import load_model, model_from_json
#===== setting for float64 for keras ======
from keras import backend as K
K.set_floatx('float64')
K.set_epsilon(1e-14)

def bpnn_2b_model(base_path, at_idx_map, Gfunc_data, s):
    """calculate sum of all atomic energies from subnetworks for dimers 
    # Arguments
        base_path: path to bpnn-2b model
        at_idx_map: a mapping between atom types and atom indexes; dictionary
        Gfunc_data: symmetry function values; 
                    dictionary with 1st layer keys = atom types,
                        2nd layer keys = atom indexes,
                        values = 2D arrays with shape=(nb_samples, nb_sym_functions)
        s: switching function
    # Returns
        bpnn 2B-energy
    """
    # scale symmetry function 
    Gfunc_scaled = {} 
    #Natom = 0
    for at_type in at_idx_map.keys():
        Gfunc_scaled[at_type] = {}

        for at in at_idx_map[at_type]:
            tmp_max = np.loadtxt(os.path.join(base_path, '2B_max_per_feature', at_type + '_max'))
            Gfunc_scaled[at_type][at] = Gfunc_data[at_type][at][:,:]/tmp_max[np.newaxis,:][0,:]
    
    # load neural network model 
    model_path = os.path.join(base_path,'model_2B_fswitch_v14s87912/34_model.json')    
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    weight_path = os.path.join(base_path,'model_2B_fswitch_v14s87912/34_weight.hdf5')
    model.load_weights(weight_path)
    
    inp_NN = [Gfunc_scaled[u][v] for u in at_idx_map.keys() for v in at_idx_map[u]] + [s]
    
    return model.predict(inp_NN).T[0] 

def bpnn_3b_model(base_path, at_idx_map, Gfunc_data, s):
    """calculate sum of all atomic energies from subnetworks for trimers 
    # Arguments
        base_path: path to bpnn-3b model
        at_idx_map: a mapping between atom types and atom indexes; dictionary
        Gfunc_data: symmetry function values; 
                    dictionary with 1st layer keys = atom types,
                        2nd layer keys = atom indexes,
                        values = 2D arrays with shape=(nb_samples, nb_sym_functions)
        s: switching function
    # Returns
        bpnn 3B-energy
    """

    # scale symmetry function 
    Gfunc_scaled = {}
    for at_type in at_idx_map.keys():
        Gfunc_scaled[at_type] = {}

        for at in at_idx_map[at_type]:
            tmp_max = np.loadtxt(os.path.join(base_path, '3B_max_per_feature', at_type + '_max'))
            Gfunc_scaled[at_type][at] = Gfunc_data[at_type][at][:,:]/tmp_max[np.newaxis,:][0,:]

    # load neural network model 
    model_path = os.path.join(base_path,'model_3B_fswitch_v16s357809/22_model.json')
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    weight_path = os.path.join(base_path,'model_3B_fswitch_v16s357809/22_weight.hdf5')
    model.load_weights(weight_path)
    
    inp_NN = [Gfunc_scaled[u][v] for u in at_idx_map.keys() for v in at_idx_map[u]] + [s]

    return model.predict(inp_NN).T[0]
    

