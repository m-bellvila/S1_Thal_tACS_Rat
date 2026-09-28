"""
init.py

Starting script to run NetPyNE-basedS1 model.

Usage:
    python init.py # Run simulation, optionally plot a raster

MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py

Contributors: salvadordura@gmail.com, fernandodasilvaborges@gmail.com, monica.bellvila@mail.utoronto.ca
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers
from netpyne import sim
import neuron
import pickle, json
import numpy as np


# cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams.py')
cfg, netParams = sim.readCmdLineArgs()

print(str(cfg.fracmorphoradius))
#print(cfg.fracmorphoradius)

sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations

## Load cells positions
with open('../data/spkTimes_v9_batch8_highgsynCT.pkl', 'rb') as fileObj: simData = pickle.load(fileObj)

cellsTags = simData['cellsTags']

# print(sim.rank,sim.net.cells[0].tags)

for i,metype in enumerate(sim.net.cells):

    if 'presyn' in metype.tags['pop']:
        ii = int(metype.tags['cellLabel'])        
        metype.tags['xnorm'] = cellsTags[ii]['xnorm']
        metype.tags['ynorm'] = cellsTags[ii]['ynorm']
        metype.tags['znorm'] = cellsTags[ii]['znorm']
        metype.tags['x'] = cellsTags[ii]['x']
        metype.tags['y'] = cellsTags[ii]['y']
        metype.tags['z'] = cellsTags[ii]['z']   

    else:
        ii2 = int(0.000001+(metype.tags['fraction']/(1/cfg.Nmorpho[metype.tags['pop']])))  

        ii = cfg.listmorphonumber[metype.tags['pop']][ii2]

        metype.tags['xnorm'] = cellsTags[ii]['xnorm']
        metype.tags['ynorm'] = cellsTags[ii]['ynorm']
        metype.tags['znorm'] = cellsTags[ii]['znorm']
        metype.tags['x'] = cellsTags[ii]['x']
        metype.tags['y'] = cellsTags[ii]['y']
        metype.tags['z'] = cellsTags[ii]['z']   

sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
sim.net.defineCellShapes()



def insert_v_ext(cell, v_ext, t_ext):
    cell.t_ext = neuron.h.Vector(t_ext)
    cell.v_ext = []
    for v in v_ext:
        cell.v_ext.append(neuron.h.Vector(v))

    # play v_ext into e_extracellular reference
    i = 0
    cell.v_ext[i].play(cell.secs['soma']['hObj'](
        0.5)._ref_e_extracellular, cell.t_ext)

# Coefficients derived from regression of potential values across depth derived from the interpolation 
#of potential artefacts in experimental electrophysiological recordings against Sim4Life electic fields across 
#4 amplitudes tested (50, 100, 200, 400uA) 
powers = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
coeffs = [ 5.86454142e-05, -4.16641407e-02,  8.98017320e+00, -4.83143435e+02,
        9.68157164e-06,  9.19035722e-03, -1.98086390e+00,  1.06572684e+02]


def make_extracellular_stimuli(cell):
    """ Function to calculate and apply external potential"""

    stimstart = cfg.ACSparams['stimstart']
    stimend = cfg.ACSparams['stimend']
    stimdif = stimend-stimstart

    # MAKING THE EXTERNAL FIELD
    n_tsteps = int(stimdif / cfg.dt + 1)
    n_start = int(stimstart/cfg.dt)
    n_end = int(stimend/cfg.dt + 1)
    t = np.arange(start=n_start, stop=n_end) * cfg.dt
    pulse = cfg.ACSparams['amp'] * \
        np.sin(2 * np.pi * cfg.ACSparams['freq'] * t / 1000)
    
    A = np.asarray(pulse.reshape(1, n_tsteps))
    y =  abs(cell.getSomaPos()[1])/1000000 #initial cell depth is in um, convert to m for calculation
    
    v_cell_ext = np.zeros((1, n_tsteps))
    
    A_b, y_b = np.broadcast_arrays(A, y)
    potential_vals = np.zeros_like(A_b, dtype = float)
    for c, (i, j) in zip(coeffs, powers):
        potential_vals += c * (A_b**i) * (y_b**j)
        
    potential_vals = potential_vals*1000*scalestim #convert V to mV, multiply by morphological scaling factor
    
    v_cell_ext[:, :] = potential_vals.reshape(1,n_tsteps)
    insert_v_ext(cell, v_cell_ext, t)
    
    return potential_vals, pulse


if cfg.ACSparams['include'] == True:
    for c,metype in enumerate(sim.net.cells):
        if 'presyn' not in metype.tags['pop']:
            if '_'.join(metype.tags['pop'].split('_',2)[0:2]) in cfg.Ipops:
                scalestim = cfg.ACSparams['Inh_scale']
                ext_field, pulse = make_extracellular_stimuli(sim.net.cells[c])
            elif '_'.join(metype.tags['pop'].split('_',2)[0:2]) in cfg.Epops:
                scalestim = cfg.ACSparams['Exc_scale']
                ext_field, pulse = make_extracellular_stimuli(sim.net.cells[c])



            

sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#
sim.analysis.plotData()         			# plot spike raster etc


# sim.analysis.plotRaster(include=cfg.recordCells, timeRange=[0,cfg.duration], orderBy='gid', orderInverse=True, labels=None, popRates=False, lw=5, marker='.', markerSize=15, figSize=(18, 12), fontSize=9, dpi=300, saveFig='../data/'+cfg.simLabel[0:9]+'/'+cfg.simLabel + '_Raster_onecellperpop.png', showFig=False)
# sim.analysis.plotRaster(timeRange=[0,cfg.duration], orderBy='gid', orderInverse=True, labels=None, popRates=False, lw=1, marker='.', markerSize=2, figSize=(18, 12), fontSize=9, dpi=300, saveFig=True, showFig=False)
# sim.analysis.plotTraces(include=cfg.recordCells, overlay=True, oneFigPer='cell', figSize=(12, 4), fontSize=7, saveFig=True)
#sim.analysis.plotTraces(include=cfg.recordCells, overlay=False, oneFigPer='trace', figSize=(18, 12), fontSize=9, saveFig=True)
# features = ['numConns','convergence']
# groups =['pop']
# for feat in features:
#    for group in groups:
#        sim.analysis.plotConn(includePre=['L1_DAC_cNA','L23_MC_cAC','L4_SS_cAD','L4_NBC_cNA','L5_TTPC2_cAD', 'L5_LBC_cNA', 'L6_TPC_L4_cAD', 'L6_LBC_cNA', 'ss_RTN_o', 'ss_RTN_m', 'ss_RTN_i', 'VPL_sTC', 'VPM_sTC', 'POm_sTC_s1'], includePost=['L1_DAC_cNA','L23_MC_cAC','L4_SS_cAD','L4_NBC_cNA','L5_TTPC2_cAD', 'L5_LBC_cNA', 'L6_TPC_L4_cAD', 'L6_LBC_cNA', 'ss_RTN_o', 'ss_RTN_m', 'ss_RTN_i', 'VPL_sTC', 'VPM_sTC', 'POm_sTC_s1'], feature=feat, groupBy=group, figSize=(24,24), saveFig=True, orderBy='gid', graphType='matrix', fontSize=18, saveData='../data/'+cfg.simLabel[0:9]+'/'+cfg.simLabel + '_' + group + '_' + feat+ '_matrix.json')

# sim.analysis.plotLFP(**{'plots': ['timeSeries'], 'electrodes': [0,1,2,3], 'timeRange': [150, cfg.duration], 'maxFreq':80, 'figSize': (16,8), 'saveFig': '../data/'+cfg.simLabel[0:9]+'/'+cfg.simLabel + '_' +'LFP1.png', 'showFig': False})
# sim.analysis.plotLFP(**{'plots': ['timeSeries'], 'electrodes': [4,5,6,7], 'timeRange': [0, 300], 'maxFreq':80, 'figSize': (16,8), 'saveFig': '../data/'+cfg.simLabel[0:9]+'/'+cfg.simLabel + '_' +'LFP2', 'showFig': False})
# sim.analysis.plotLFP(**{'plots': ['timeSeries'], 'electrodes': [8,9,10,11], 'timeRange': [0, 300], 'maxFreq':80, 'figSize': (16,8), 'saveFig': '../data/'+cfg.simLabel[0:9]+'/'+cfg.simLabel + '_' +'LFP3', 'showFig': False})
